import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from config import Seg
from data import get_files
import segmentation_models_pytorch as smp


# --- Inference Dataset (no masks) ---
class InferenceDataset(Dataset):
    def __init__(self, image_paths, normalize=True):
        self.image_paths = image_paths
        self.normalize = normalize
        self.target_height = 256
        self.target_width = 320

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[:, :, ::-1].astype(np.float32)  # BGR to RGB
        if self.normalize:
            image /= 255.0
        image = cv2.resize(image, (self.target_width, self.target_height))
        image = np.transpose(image, (2, 0, 1))  # (C, H, W)
        return torch.tensor(image, dtype=torch.float32), self.image_paths[idx]


# --- Load model checkpoint ---
def load_model(checkpoint_path, device):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()
    return model


# --- Apply binary mask to image (keep tongue only) ---
def apply_mask(image, mask):
    mask = mask.astype(np.uint8)
    mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    mask = (mask > 0.5).astype(np.uint8)
    mask_3ch = np.stack([mask] * 3, axis=-1)
    return np.where(mask_3ch == 1, image, 0)


# --- Inference on PNGs from data/scores/train, save to results/samples ---
def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    image_dir = "data/scores"
    image_paths = get_files(image_dir, extensions=["*.png"])
    if len(image_paths) == 0:
        raise FileNotFoundError(f"❌ No .png files found in: {image_dir}")

    dataset = InferenceDataset(image_paths)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    checkpoint_path = os.path.join(Seg.model_dir, "deeplabv3plus_best.pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"❌ Checkpoint not found at: {checkpoint_path}")

    model = load_model(checkpoint_path, device)
    print(f"✔️ Loaded model from {checkpoint_path}")

    os.makedirs(Seg.samples, exist_ok=True)

    for img_tensor, img_path in tqdm(loader, desc="Inference"):
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            pred = model(img_tensor)
            pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()

        pred_mask_bin = (pred_mask > 0.5).astype(np.uint8)
        original = cv2.imread(img_path[0])[:, :, ::-1]  # BGR to RGB
        result = apply_mask(original, pred_mask_bin)

        filename = os.path.basename(img_path[0]).rsplit(".", 1)[0]
        save_path = os.path.join(Seg.samples, f"{filename}_seg.png")
        cv2.imwrite(save_path, result[:, :, ::-1])  # RGB to BGR

    print(f"✅ Saved segmented outputs to: {Seg.samples}")


if __name__ == "__main__":
    test()
