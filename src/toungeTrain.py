import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import efficientnet_b1, EfficientNet_B1_Weights
import segmentation_models_pytorch as smp
from tqdm import tqdm
from config import Seg
from data import load_score_data

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load segmentation model
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

seg_model = load_seg_model(os.path.join(Seg.model_dir, "deeplabv3plus_best.pth"))

# Load data
train_loader, val_loader = load_score_data(seg_model, device)

# Define regression model using EfficientNet-B1
model = efficientnet_b1(weights=EfficientNet_B1_Weights.DEFAULT)
model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 5)  # ✅ match previous output size
)

model.to(device)

# Load best model if exists
checkpoint_path = "results/scoreCheckpoints/best_model.pth"
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"✅ Loaded checkpoint from {checkpoint_path}")

# Loss, Optimizer, Scheduler
criterion = nn.SmoothL1Loss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4)

# Training loop
num_epochs = 500
best_val_loss = float('inf')
os.makedirs("results/scoreCheckpoints", exist_ok=True)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images, targets = images.to(device), targets.to(device)
            outputs = model(images)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")

    scheduler.step(avg_val_loss)

    # Save best
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), checkpoint_path)
        print("✅ Best model saved.")
