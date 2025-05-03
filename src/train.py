import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import segmentation_models_pytorch as smp

from data import get_files, load_segmentation_dataset, SegmentationDataset
from config import Seg  # Seg must have .images, .masks, .batch, .model_dir, .epochs, .lr

# ------------------------ Metrics & Loss ------------------------ #
def compute_metrics(preds, targets, threshold=0.5, eps=1e-7):
    p = (preds > threshold).astype(np.uint8)
    t = (targets > threshold).astype(np.uint8)
    tp = np.sum(p & t)
    fp = np.sum(p & ~t)
    fn = np.sum(~p & t)
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)
    return {"iou": iou, "dice": dice}

def dice_loss(pred, target, smooth=1e-6):
    pred = torch.sigmoid(pred).view(pred.size(0), -1)
    target = target.view(target.size(0), -1)
    inter = (pred * target).sum(dim=1)
    return 1 - ((2 * inter + smooth) /
                (pred.sum(dim=1) + target.sum(dim=1) + smooth)).mean()

# ------------------------ Training & Validation ------------------------ #
def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for imgs, masks in tqdm(loader, desc="Train"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        if out.shape[-2:] != masks.shape[-2:]:
            out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
        loss = criterion(out, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss, all_preds, all_targets = 0.0, [], []
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validate"):
            imgs, masks = imgs.to(device), masks.to(device)
            out = model(imgs)
            if out.shape[-2:] != masks.shape[-2:]:
                out = F.interpolate(out, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            total_loss += criterion(out, masks).item()
            all_preds.append(torch.sigmoid(out).cpu().numpy())
            all_targets.append(masks.cpu().numpy())
    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    metrics = compute_metrics(preds, targets)
    return total_loss / len(loader), metrics

# ------------------------ Main ------------------------ #
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Only look for .bmp files
    image_paths = get_files(Seg.images, extensions=["*.bmp"])
    mask_paths = get_files(Seg.masks, extensions=["*.bmp"])
    assert len(image_paths) == len(mask_paths), "Image/mask count mismatch"
    print(f"Found {len(image_paths)} .bmp images")

    train_imgs, val_imgs, train_masks, val_masks = load_segmentation_dataset(image_paths, mask_paths)
    train_loader = DataLoader(SegmentationDataset(train_imgs, train_masks), batch_size=Seg.batch, shuffle=True, num_workers=4)
    val_loader = DataLoader(SegmentationDataset(val_imgs, val_masks), batch_size=Seg.batch, shuffle=False, num_workers=4)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None
    ).to(device)

    # Freeze encoder for fine-tuning
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=Seg.lr)
    criterion = dice_loss

    best_iou = 0.0
    for epoch in range(1, Seg.epochs + 1):
        print(f"\nEpoch {epoch}/{Seg.epochs}")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, metrics = validate(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Val IoU: {metrics['iou']:.4f}, Dice: {metrics['dice']:.4f}")

        if metrics["iou"] > best_iou:
            best_iou = metrics["iou"]
            os.makedirs(Seg.model_dir, exist_ok=True)
            ckpt_path = os.path.join(Seg.model_dir, "deeplabv3plus_best.pth")
            torch.save(model.state_dict(), ckpt_path)
            print(f"✔️ Saved best model to: {ckpt_path}")

if __name__ == "__main__":
    main()
