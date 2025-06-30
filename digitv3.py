import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from lion_pytorch import Lion
from PIL import Image
import time

# 🔧 Excitation-based dropout (safe for FC layers)
class ExcitationDropout(nn.Module):
    def __init__(self, temperature=0.7):
        super().__init__()
        self.temperature = temperature

    def forward(self, x):
        if not self.training:
            return x
        prob = torch.sigmoid(x / self.temperature)
        drop_mask = torch.bernoulli(1 - prob).to(x.device)
        return x * drop_mask

# 🧱 Deformable conv block (no dropout here)
class DeformableConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.offset = nn.Conv2d(in_ch, 2 * kernel_size * kernel_size, kernel_size=3, padding=1)
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, kernel_size, kernel_size) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_ch))

    def forward(self, x):
        offset = self.offset(x)
        return deform_conv2d(x, offset, self.weight, self.bias, padding=1)

# 🧠 Main model
class DeformableMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = DeformableConvBlock(1, 32)
        self.block2 = DeformableConvBlock(32, 64)
        self.pool = nn.AdaptiveAvgPool2d((14, 14))  # Preserves structure, smooth gradient

        self.classifier = nn.Sequential(
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            ExcitationDropout(temperature=0.7),  # Safe, light regularization
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.pool(F.relu(self.block1(x)))
        x = self.pool(F.relu(self.block2(x)))
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 🏋️ Training function
def train():
    print("Setting up...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeformableMNIST().to(device)
    optimizer = Lion(model.parameters(), lr=1e-3, weight_decay=1e-2)
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
        start_time = time.time()
        total_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")
        print(f"Epoch {epoch + 1} done in {time.time() - start_time:.1f} sec. Total Loss: {total_loss:.4f}")

    torch.save(model.state_dict(), "deformable_mnist_final.pth")
    print("✅ Model saved to deformable_mnist_final.pth")

# 🔍 Inference
def classify_digit(image_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeformableMNIST().to(device)
    model.load_state_dict(torch.load("deformable_mnist_final.pth", map_location=device))
    model.eval()

    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)  # [1, 1, 28, 28]
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        output = model(image_tensor)
        return output.argmax(dim=1).item()

# 🔧 Example usage
if __name__ == "__main__":
    # Uncomment to train:
     train()

    # Example classification
    # img = Image.open("some_digit.png").convert("L").resize((28, 28))
    # tensor = transforms.ToTensor()(img)
    # print("Predicted digit:", classify_digit(tensor))
     # pass
