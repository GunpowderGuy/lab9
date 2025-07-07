import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Cargar el modelo
from model import CNN  # Asegúrate que este es el modelo correcto
model = CNN()
model.load_state_dict(torch.load("model_final.pth", map_location="cpu"))
model.eval()

# Dataset de validación
val_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

val_dataset = datasets.ImageFolder("tiny-imagenet-200/val", transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Evaluación
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Matriz de confusión
cm = confusion_matrix(all_labels, all_preds)
acc = accuracy_score(all_labels, all_preds)

# Visualización
plt.figure(figsize=(12, 10))
sns.heatmap(cm, cmap='Blues', cbar=False)
plt.title(f'Matriz de Confusión - Accuracy: {acc:.2%}')
plt.xlabel('Predicción')
plt.ylabel('Etiqueta Real')
plt.savefig("confusion_matrix.png")
plt.show()
