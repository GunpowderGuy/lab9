import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.ops import deform_conv2d
import torchvision
import torch.optim as optim

# ==== Configuration ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "./tiny-imagenet-200"
BATCH_SIZE = 64
EPOCHS = 5
LR = 1e-3
MODEL_PATH = "deformable_tiny_imagenet.pth"
NUM_CLASSES = 200

# ==== Transforms & Data Loaders ====
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                         std=[0.2302, 0.2265, 0.2262]),
])

train_set = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=transform)
val_set = datasets.ImageFolder(os.path.join(DATA_DIR, "val", "images"), transform=transform)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ==== ExcitationDropout (defined inline) ====
class ExcitationDropout(nn.Module):
    def __init__(self, base_p=0.4):
        super().__init__()
        self.base_p = base_p

    def forward(self, x):
        if not self.training:
            return x
        prob = torch.sigmoid(x)
        mask = (torch.rand_like(x) < (1 - prob * self.base_p)).float()
        # scale to keep expected magnitude
        return x * mask / (1 - self.base_p)

# ==== Deformable CNN Model ====
class DeformableTinyImageNet(nn.Module):
    def __init__(self):
        super().__init__()
        # First deformable block
        self.offset1 = nn.Conv2d(3,    18, kernel_size=3, padding=1)
        self.conv1   = nn.Conv2d(3,   64, kernel_size=3, padding=1)  # linear weights
        # Second deformable block
        self.offset2 = nn.Conv2d(64,  2*3*3, kernel_size=3, padding=1)
        self.conv2   = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)

        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(128, 256)
        self.drop = ExcitationDropout(base_p=0.4)
        self.fc2 = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        offset = self.offset1(x)
        x = deform_conv2d(x, offset, self.conv1.weight, self.conv1.bias, padding=1)
        x = F.relu(self.bn1(x))
        x = F.dropout(x, p=0.2)

        offset = self.offset2(x)
        x = deform_conv2d(x, offset, self.conv2.weight, self.conv2.bias, padding=1)
        x = F.relu(self.bn2(x))
        x = F.dropout(x, p=0.3)

        x = self.pool(x).flatten(1)

        x = F.relu(self.fc1(x))
        x = self.drop(x)
        return self.fc2(x)

model = DeformableTinyImageNet().to(DEVICE)

optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()

# ==== Training Loop ====
def train():
    print("Training on", DEVICE)
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for i, (imgs, labels) in enumerate(train_loader, 1):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            if i % 100 == 0 or i == len(train_loader):
                print(f"Epoch {epoch} [{i}/{len(train_loader)}]  loss: {loss.item():.4f}")

        print(f"Epoch {epoch} complete — time: {time.time()-t0:.1f}s  avg loss: {epoch_loss/len(train_loader):.4f}")
        torch.save(model.state_dict(), MODEL_PATH)
        print(f"Saved checkpoint to {MODEL_PATH}")

# ==== Inference Function ====
def classify_image(img_tensor):
    model.eval()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    with torch.no_grad():
        img_tensor = img_tensor.to(DEVICE).unsqueeze(0)
        logits = model(img_tensor)
        pred = logits.argmax(dim=1).item()
        class_name = train_set.classes[pred]
        return pred, class_name

if __name__ == "__main__":
    train()
