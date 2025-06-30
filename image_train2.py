import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import DeformConv2d
import torch.nn.functional as F
from lion_pytorch import Lion  # asegúrate de tener instalado lion-pytorch

# === Configuración general ===
DATA_DIR = './tiny-imagenet-200'
BATCH_SIZE = 64
EPOCHS = 10
NUM_CLASSES = 200
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === Dataset y DataLoader ===
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)
val_img_dir = os.path.join(DATA_DIR, 'val', 'images')
val_set = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform)  # usamos train como workaround
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

# === Módulo de convolución deformable ===
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

# === Modelo ===
class TinyImageNetCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = DeformConvBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DeformConvBlock(64, 128)
        self.pool2 = nn.AdaptiveAvgPool2d((4, 4))

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.drop1 = nn.Dropout(p=0.5)  # regularización fuerte para FC
        self.fc2 = nn.Linear(512, NUM_CLASSES)

    def forward(self, x):
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = torch.flatten(x, 1)
        x = self.drop1(F.relu(self.fc1(x)))
        return self.fc2(x)

# === Entrenamiento ===
def train():
    model = TinyImageNetCNN().to(DEVICE)
    optimizer = Lion(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        model.train()
        print(f"=== Epoch {epoch + 1}/{EPOCHS} ===")
        total_loss, correct, total = 0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

        print(f"Train Loss: {total_loss / len(train_loader):.4f}, Acc: {100 * correct / total:.2f}%")

    torch.save(model.state_dict(), 'model_tiny_imagenet.pth')
    print("Model saved as model_tiny_imagenet.pth")

# === Inferencia ===
def classify_image(image_path):
    from PIL import Image
    model = TinyImageNetCNN().to(DEVICE)
    model.load_state_dict(torch.load('model_tiny_imagenet.pth', map_location=DEVICE))
    model.eval()

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = outputs.max(1)
    print(f"Predicted class: {predicted.item()}")

if __name__ == '__main__':
    train()
    # classify_image('tiny-imagenet-200/val/images/val_0.JPEG')  # ejemplo de uso

