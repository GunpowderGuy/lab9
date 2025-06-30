import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import DeformConv2d
import torch.nn.functional as F
from lion_pytorch import Lion
from PIL import Image

class ExcitationDropout(nn.Module):
    """
    Element-wise excitation dropout:
    Drops activations with probability proportional to sigmoid(x) * base_p.
    """
    def __init__(self, base_p: float = 0.5):
        super().__init__()
        self.base_p = base_p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.base_p == 0.0:
            return x
        # Compute drop probability per element
        p_drop = self.base_p * torch.sigmoid(x)
        p_keep = 1.0 - p_drop
        # Sample mask and scale
        mask = torch.rand_like(x) < p_keep
        return x * mask.to(x.dtype) / p_keep



# === Config ===
DATA_DIR = './tiny-imagenet-200'
BATCH_SIZE = 64
EPOCHS = 10
NUM_CLASSES = 200
SAVE_EVERY = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.backends.cudnn.benchmark = True
print(f'Using device: {DEVICE}')

# === Data ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)  # workaround
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# === Deformable Block ===
class DeformConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        offset = self.offset_conv(x)
        x = self.deform_conv(x, offset)
        x = self.bn(x)
        return F.relu(x)

# === Model ===
class TinyImageNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DeformConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DeformConvBlock(64, 128)
        self.pool2 = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(128, 512)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.drop1(F.relu(self.fc1(x)))
        return self.fc2(x)

# === Train ===
def train():
    model = TinyImageNetCNN().to(DEVICE)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    scaler = torch.cuda.amp.GradScaler() if DEVICE.type == 'cuda' else None
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        print(f"\n=== Epoch {epoch + 1}/{EPOCHS} ===")
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
                outputs = model(images)
                loss = criterion(outputs, labels)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        acc = 100 * correct / total
        print(f"Train Loss: {total_loss / len(train_loader):.4f}, Accuracy: {acc:.2f}%")

        # === Validation ===
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(labels).sum().item()
                val_total += labels.size(0)

        val_acc = 100 * val_correct / val_total
        print(f"Val Loss: {val_loss / len(val_loader):.4f}, Val Accuracy: {val_acc:.2f}%")

        # === Periodic Save ===
        if (epoch + 1) % SAVE_EVERY == 0:
            torch.save(model.state_dict(), f'model_epoch{epoch+1}.pth')
            print(f"Saved model at epoch {epoch+1}")

    torch.save(model.state_dict(), 'model_final.pth')
    print("Final model saved.")

# === Inference ===
def classify_image(image_path):
    model = TinyImageNetCNN().to(DEVICE)
    model.load_state_dict(torch.load('model_final.pth', map_location=DEVICE))
    model.eval()
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    print(f"Predicted class: {predicted.item()}")

if __name__ == '__main__':
    train()
    # classify_image('tiny-imagenet-200/val/images/val_0.JPEG')
