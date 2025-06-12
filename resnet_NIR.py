import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.metrics import classification_report

# Dataset path
DATASET_PATH = r"E:\RGB-NIR\nirscene1"

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize for ResNet
    transforms.Normalize([0.5], [0.5])  # Normalize single-channel NIR images
])

# Custom dataset class
class NIR_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            nir_images = [f for f in os.listdir(class_path) if f.endswith("_nir.tiff")]

            for nir_file in nir_images:
                self.data.append((os.path.join(class_path, nir_file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        nir_path, label = self.data[idx]
        nir_image = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)

        if self.transform:
            nir_image = self.transform(nir_image)
            nir_image = nir_image.expand(3, -1, -1)  # Convert single-channel to 3-channel

        return nir_image, label

# Create train/test datasets
dataset = NIR_Dataset(DATASET_PATH, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load Pretrained ResNet
model = resnet18(pretrained=True)

# Modify final layer for 9 classes
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 9)

# Move to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}")

# Evaluate Model & Generate Class-Wise Metrics
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Compute class-wise accuracy, precision, recall, F1-score
class_names = dataset.classes
report = classification_report(all_labels, all_preds, target_names=class_names)
print(report)
