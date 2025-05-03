import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

from config import Seg
from data import get_files


class SimpleDataset(Dataset):
    def __init__(self, image_paths, normalize=True):
        self.image_paths = image_paths
        self.normalize = normalize
        self.target_size = (320, 256)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = cv2.imread(path)[:, :, ::-1].astype(np.float32)
        if self.normalize:
            img /= 255.0
        img = cv2.resize(img, self.target_size)
        img = np.transpose(img, (2, 0, 1))
        return torch.tensor(img, dtype=torch.float32), path


def load_deeplabv3plus(path, device):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(path, map_location=device))
    return model.to(device).eval()


def apply_mask(mask):
    return (cv2.resize(mask, (320, 256)) > 0.5).astype(np.uint8)


def extract_raw_features(seg_img):
    gray = cv2.cvtColor(seg_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(seg_img, cv2.COLOR_BGR2HSV)
    total_area = gray.shape[0] * gray.shape[1]
    _, binary_mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    # Coated region: white or yellow
    coated_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 0, 200), (180, 30, 255)),
        cv2.inRange(hsv, (15, 60, 180), (45, 255, 255))
    )
    coated_ratio = np.sum(coated_mask > 0) / total_area

    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        hull = cv2.convexHull(cnt)
        jaggedness = cv2.arcLength(cnt, True) / (cv2.arcLength(hull, True) + 1e-5)
    else:
        jaggedness = 1.0

    clahe = cv2.createCLAHE(clipLimit=2.0).apply(gray)
    edges = cv2.Canny(clahe, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    crack_count = sum(1 for c in contours if cv2.arcLength(c, False) > 30)

    papillae_texture = np.std(cv2.Laplacian(gray, cv2.CV_64F))

    red_mask = cv2.bitwise_or(
        cv2.inRange(hsv, (0, 70, 50), (10, 255, 255)),
        cv2.inRange(hsv, (160, 70, 50), (180, 255, 255))
    )
    red_ratio = np.sum(red_mask > 0) / total_area

    return {
        "Coated_Tongue": coated_ratio,
        "Jagged_Shape": jaggedness,
        "Cracks": crack_count,
        "Filiform_Papillae": papillae_texture,
        "Fungiform_Redness": red_ratio
    }, coated_mask, edges, red_mask


def normalize(value, key, stats):
    min_v, max_v = stats[key]["min"], stats[key]["max"]
    return float(np.clip((value - min_v) / (max_v - min_v + 1e-6) * 10, 0, 10))


def overlay_mask_on_image(img, mask, color, alpha=0.4):
    overlay = img.copy()
    mask = mask.astype(bool)

    # Ensure the color is an array of shape (N, 3) to match img[mask]
    color_array = np.zeros_like(img, dtype=np.uint8)
    color_array[:, :] = color  # broadcast the color

    overlay[mask] = cv2.addWeighted(img[mask], 1 - alpha, color_array[mask], alpha, 0)
    return overlay


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    image_paths = get_files("data/scores", extensions=["*.png", "*.jpg", "*.bmp"])
    if not image_paths:
        raise FileNotFoundError("❌ No images found in data/scores")

    dataset = SimpleDataset(image_paths)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    model_path = os.path.join(Seg.model_dir, "deeplabv3plus_best.pth")
    model = load_deeplabv3plus(model_path, device)
    print(f"[INFO] Model loaded from {model_path}")

    os.makedirs(Seg.scored, exist_ok=True)

    raw_data = []

    for img_tensor, img_path in tqdm(loader, desc="Segmenting"):
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.sigmoid(output).squeeze().cpu().numpy()
            mask = apply_mask(pred)

        orig = cv2.imread(img_path[0])
        if orig is None:
            continue
        orig = cv2.resize(orig, (mask.shape[1], mask.shape[0]))
        tongue_only = np.where(mask[:, :, None] == 1, orig, 0)

        features, coated_mask, cracks_mask, red_mask = extract_raw_features(tongue_only)

        raw_data.append({
            "file_name": os.path.basename(img_path[0]),
            "file_path": img_path[0],
            "raw_features": features,
            "mask": mask,
            "image": tongue_only,
            "coated_mask": coated_mask,
            "cracks_mask": cracks_mask,
            "red_mask": red_mask
        })

    feature_stats = {
        key: {
            "min": float(np.min([d["raw_features"][key] for d in raw_data])),
            "max": float(np.max([d["raw_features"][key] for d in raw_data]))
        }
        for key in raw_data[0]["raw_features"]
    }

    feature_colors = {
        "Coated_Tongue": (0, 255, 255),
        "Jagged_Shape": (0, 0, 255),  # Not localized
        "Cracks": (255, 0, 0),
        "Filiform_Papillae": (0, 255, 0),  # Not localized
        "Fungiform_Redness": (255, 0, 255)
    }

    results = []

    for entry in raw_data:
        scores = {k: normalize(v, k, feature_stats) for k, v in entry["raw_features"].items()}
        result_img = entry["image"].copy()

        if scores["Coated_Tongue"] >= 1:
            result_img = overlay_mask_on_image(result_img, entry["coated_mask"], feature_colors["Coated_Tongue"])

        if scores["Cracks"] >= 1:
            result_img = overlay_mask_on_image(result_img, entry["cracks_mask"], feature_colors["Cracks"])

        if scores["Fungiform_Redness"] >= 1:
            result_img = overlay_mask_on_image(result_img, entry["red_mask"], feature_colors["Fungiform_Redness"])

        out_path = os.path.join(Seg.scored, entry["file_name"])
        cv2.imwrite(out_path, result_img)

        results.append({
            "file_name": entry["file_name"],
            "file_path": entry["file_path"],
            "scores": scores
        })

    with open(Seg.scores, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[✅] Saved results to: {Seg.scored}")
