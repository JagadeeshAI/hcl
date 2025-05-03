import os
import json
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import segmentation_models_pytorch as smp
from sklearn.model_selection import train_test_split

from config import Seg
from data import get_files

# --- Dataset with Segmentation and Score Loading ---
class TongueSegScoreDataset(Dataset):
    def __init__(self, image_paths, score_dict, transform=None):
        self.image_paths = image_paths
        self.score_dict = score_dict
        self.transform = transform
        self.target_height = 256
        self.target_width = 320

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)

        # Load and preprocess image
        image = cv2.imread(img_path)[:, :, ::-1].astype(np.float32) / 255.0
        image = cv2.resize(image, (self.target_width, self.target_height))
        image_tensor = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32)

        # Segment image
        with torch.no_grad():
            seg_pred = seg_model(image_tensor.unsqueeze(0).to(device))
            seg_mask = torch.sigmoid(seg_pred).squeeze().cpu().numpy()
        seg_bin = (seg_mask > 0.5).astype(np.uint8)
        mask_3ch = np.stack([seg_bin] * 3, axis=-1)
        segmented_img = np.where(mask_3ch == 1, image, 0)

        if self.transform:
            segmented_img = self.transform(Image.fromarray((segmented_img * 255).astype(np.uint8)))

        # Get ground truth scores
        scores = torch.tensor([
            self.score_dict[filename]["Coated_Tongue"],
            self.score_dict[filename]["Jagged_Shape"],
            self.score_dict[filename]["Cracks"],
            self.score_dict[filename]["Filiform_Papillae"],
            self.score_dict[filename]["Fungiform_Redness"]
        ], dtype=torch.float32)

        return segmented_img, scores

# --- Load segmentation model ---
def load_seg_model(checkpoint_path):
    model = smp.DeepLabV3Plus(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    return model.to(device).eval()

# --- Load score annotations ---
with open("/media/jag/volD/hcl/results/scores.json", "r") as f:
    raw_scores = json.load(f)
    score_dict = {entry["file_name"]: entry["scores"] for entry in raw_scores}

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_dir = "data/scores"
image_paths = get_files(image_dir, extensions=["*.png"])

# Train/Val Split
train_paths, val_paths = train_test_split(image_paths, test_size=0.2, random_state=42)

# Load segmentation model
seg_model = load_seg_model(os.path.join(Seg.model_dir, "deeplabv3plus_best.pth"))

# Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Datasets and loaders
train_dataset = TongueSegScoreDataset(train_paths, score_dict, transform=transform)
val_dataset = TongueSegScoreDataset(val_paths, score_dict, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# --- Regression model ---
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, 5)
)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Training loop ---
num_epochs = 10
best_val_loss = float('inf')
os.makedirs("results/scoreCheckpoints", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "results/scoreCheckpoints/best_model.pth")
        print("✅ Best model saved.")