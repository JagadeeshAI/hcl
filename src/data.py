import os
import cv2
import glob
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from PIL import Image
from config import Seg  

# --- Utility to Get Files ---
def get_files(path_pattern, extensions=None):
    if extensions is None:
        extensions = ['*.bmp', '*.jpg', '*.png']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(path_pattern, ext)))
    return sorted(files)

# --- Segmentation Dataset ---
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, normalize=True, resize=(256, 320)):
        assert len(image_paths) == len(mask_paths), "Image and mask count mismatch"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.normalize = normalize
        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.normalize:
            image /= 255.0
            mask /= 255.0

        if self.resize:
            image = cv2.resize(image, self.resize)
            mask = cv2.resize(mask, self.resize, interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (2, 0, 1))
        mask = np.expand_dims(mask, axis=0)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# --- Score Prediction Dataset with Segmentation ---
class TongueSegScoreDataset(Dataset):
    def __init__(self, image_paths, score_dict, seg_model, device, transform=None, resize=(256, 320)):
        self.image_paths = image_paths
        self.score_dict = score_dict
        self.seg_model = seg_model
        self.device = device
        self.transform = transform
        self.resize = resize

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        image = cv2.imread(img_path)[:, :, ::-1].astype(np.float32) / 255.0
        image = cv2.resize(image, self.resize)
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        with torch.no_grad():
            seg_pred = self.seg_model(image_tensor.unsqueeze(0).to(self.device))
            seg_mask = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
        seg_bin = (seg_mask > 0.5).astype(np.uint8)
        mask_3ch = np.stack([seg_bin]*3, axis=-1)
        segmented_img = np.where(mask_3ch == 1, image, 0)

        if self.transform:
            segmented_img = self.transform(Image.fromarray((segmented_img * 255).astype(np.uint8)))

        scores = torch.tensor([
            self.score_dict[filename]["Coated_Tongue"],
            self.score_dict[filename]["Jagged_Shape"],
            self.score_dict[filename]["Cracks"],
            self.score_dict[filename]["Filiform_Papillae"],
            self.score_dict[filename]["Fungiform_Redness"]
        ], dtype=torch.float32)

        return segmented_img, scores

# --- Loader Helpers ---
def load_segmentation_data():
    image_paths = get_files(Seg.images)
    mask_paths = get_files(Seg.masks)
    train_x, val_x, train_y, val_y = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=42
    )
    train_dataset = SegmentationDataset(train_x, train_y)
    val_dataset = SegmentationDataset(val_x, val_y)
    return (
        DataLoader(train_dataset, batch_size=Seg.batch, shuffle=True),
        DataLoader(val_dataset, batch_size=Seg.batch, shuffle=False)
    )

def load_score_data(seg_model, device):
    image_paths = get_files("data/scores", extensions=["*.png"])
    with open("/media/jag/volD/hcl/results/scores.json", "r") as f:
        raw_scores = json.load(f)
        score_dict = {entry["file_name"]: entry["scores"] for entry in raw_scores}

    train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TongueSegScoreDataset(train_paths, score_dict, seg_model, device, transform=transform)
    val_dataset = TongueSegScoreDataset(val_paths, score_dict, seg_model, device, transform=transform)

    return (
        DataLoader(train_dataset, batch_size=8, shuffle=True),
        DataLoader(val_dataset, batch_size=8, shuffle=False)
    )
