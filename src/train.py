import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import PneumoniaDataset, get_transforms
from model import get_model
from tqdm import tqdm import tqdm
import os

# Kaggle dataset: https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
data_dir = "data/chest_xray"

train_dataset = PneumoniaDataset(os.path.join(data_dir, "train"), transform=get_transforms(train=True))
val_dataset   = PneumoniaDataset(os.path.join(data_dir, "val"),   transform=get_transforms(train=False))
test_dataset  = PneumoniaDataset(os.path.join(data_dir, "test"),  transform=get_transforms(train=False))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False, num_workers=4)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = get_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

best_acc = 0.0
for epoch in range(15):
    model.train()
    running_loss = 0.0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/15"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch+1} - Val Accuracy: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "models/best_pneumonia_model.pth")

    scheduler.step()

print(f"Training finished! Best validation accuracy: {best_acc:.2f}%")