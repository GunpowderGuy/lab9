import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import time

# Deformable Conv Block
class DeformableConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.offset = nn.Conv2d(in_ch, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, x):
        offset = self.offset(x)
        return deform_conv2d(x, offset, self.weight, self.bias, padding=self.padding)

# Deformable CNN for MNIST
class DeformableMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DeformableConvBlock(1, 32)
        self.block2 = DeformableConvBlock(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.block1(x)))
        x = self.pool(F.relu(self.block2(x)))
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Training loop with feedback
def train():
    print("Setting up...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeformableMNIST().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.ToTensor()])
    train_loader = DataLoader(
        datasets.MNIST(root='.', train=True, download=True, transform=transform),
        batch_size=64, shuffle=True
    )

    print("Starting training on", device)
    model.train()
    for epoch in range(3):
        print(f"\n=== Epoch {epoch + 1} ===")
        epoch_start = time.time()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        duration = time.time() - epoch_start
        print(f"Epoch {epoch + 1} done in {duration:.1f} sec. Total Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train()
