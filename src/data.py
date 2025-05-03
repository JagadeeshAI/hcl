import os
import cv2
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from config import Seg  # Make sure Seg has .images, .masks, .batch

# Get image or mask files
def get_files(path_pattern, extensions=None):
    if extensions is None:
        extensions = ['*.bmp', '*.Bmp', '*.jpg', '*.JPG', '*.png', '*.PNG']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(path_pattern, ext)))
    return sorted(files)

# Segmentation Dataset class
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, normalize=True):
        assert len(image_paths) == len(mask_paths), "Image and mask count mismatch"
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.normalize = normalize
        self.target_height = 256
        self.target_width = 320

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])[:, :, ::-1].astype(np.float32)  # BGR to RGB
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE).astype(np.float32)

        if self.normalize:
            image /= 255.0
            mask /= 255.0  # normalize mask (depends on model setup)

        image = cv2.resize(image, (self.target_width, self.target_height))
        mask = cv2.resize(mask, (self.target_width, self.target_height), interpolation=cv2.INTER_NEAREST)

        image = np.transpose(image, (2, 0, 1))  # (C, H, W)
        mask = np.expand_dims(mask, axis=0)    # (1, H, W)

        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# Load paired image/mask paths and split
def load_segmentation_dataset(image_paths, mask_paths, random_state=360):
    image_paths = sorted(image_paths)
    mask_paths = sorted(mask_paths)

    train_x, val_x, train_y, val_y = train_test_split(
        image_paths, mask_paths, test_size=0.15, random_state=random_state, shuffle=True
    )
    return train_x, val_x, train_y, val_y

# Optional: standalone test
def main():
    image_paths = get_files(Seg.images, extensions=["*.bmp"])
    mask_paths = get_files(Seg.masks, extensions=["*.bmp"])

    print(f"Total images: {len(image_paths)}, Total masks: {len(mask_paths)}")
    if len(image_paths) == 0 or len(mask_paths) == 0:
        print("❌ No image or mask files found.")
        return

    train_images, val_images, train_masks, val_masks = load_segmentation_dataset(image_paths, mask_paths)

    print(f"Train: {len(train_images)} images, Val: {len(val_images)} images")

    train_dataset = SegmentationDataset(train_images, train_masks)
    val_dataset = SegmentationDataset(val_images, val_masks)

    train_loader = DataLoader(train_dataset, batch_size=Seg.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=Seg.batch, shuffle=False, num_workers=4)

    x_train, y_train = next(iter(train_loader))
    x_val, y_val = next(iter(val_loader))

    print(f"Training batch image shape: {x_train.shape}")
    print(f"Training batch mask shape: {y_train.shape}")   
    print(f"Validation batch image shape: {x_val.shape}")
    print(f"Validation batch mask shape: {y_val.shape}")

if __name__ == "__main__":
    main()
