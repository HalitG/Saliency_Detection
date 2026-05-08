import torch
import torch.nn as nn
import torch.optim as optim

# Import your custom modules from your other files!
from sod_model import SaliencyBaseline
from data_loader import get_dataloaders 

# 1. Setup Device and Data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, val_loader, test_loader = get_dataloaders('data', batch_size=32)

# 2. Initialize Model
model = SaliencyBaseline().to(device)

# 3. Define Loss and Optimizer
def get_iou(pred, target):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection
    return (intersection + 1e-6) / (union + 1e-6)

# Loss function: Binary Cross-Entropy + 0.5 * (1 - IoU)
def hybrid_loss(pred, target):
    bce = nn.BCELoss()(pred, target)
    iou = get_iou(pred, target)
    return bce + 0.5 * (1 - iou)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# 4. Training setup
epochs = 25
best_val_loss = float('inf')
patience = 3 
no_improvement_count = 0

# 5. The Training Loop
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = hybrid_loss(outputs, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            val_outputs = model(imgs)
            val_loss += hybrid_loss(val_outputs, masks).item()
    
    avg_train_loss = train_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1:02d}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
    
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        no_improvement_count = 0
    else:
        no_improvement_count += 1

    # Training epochs: 15-25 (early stop when validation loss stops improving)
    if (epoch + 1) >= 15 and no_improvement_count >= patience:
        print(f"Early stopping triggered at epoch {epoch+1}.")
        break