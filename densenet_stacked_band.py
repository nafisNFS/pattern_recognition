import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import densenet121
from sklearn.metrics import classification_report

# Dataset path
DATASET_PATH = r"E:\RGB-NIR\nirscene1"

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize for DenseNet
    transforms.Normalize([0.5] * 4, [0.5] * 4)  # Normalize 4 channels
])

# Custom dataset class
class RGBNIR_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            rgb_images = [f for f in os.listdir(class_path) if f.endswith("_rgb.tiff")]

            for rgb_file in rgb_images:
                nir_file = rgb_file.replace("_rgb.tiff", "_nir.tiff")
                if os.path.exists(os.path.join(class_path, nir_file)):
                    self.data.append((os.path.join(class_path, rgb_file),
                                      os.path.join(class_path, nir_file),
                                      label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, nir_path, label = self.data[idx]
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)
        nir_image = cv2.imread(nir_path, cv2.IMREAD_UNCHANGED)

        if rgb_image.shape[:2] != nir_image.shape[:2]:
            nir_image = cv2.resize(nir_image, (rgb_image.shape[1], rgb_image.shape[0]))

        nir_image = np.expand_dims(nir_image, axis=-1)  # Shape: (H, W, 1)
        combined_image = np.concatenate((rgb_image, nir_image), axis=-1)  # Shape: (H, W, 4)

        if self.transform:
            combined_image = self.transform(combined_image)

        return combined_image, label

# Create train/test datasets
dataset = RGBNIR_Dataset(DATASET_PATH, transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Load Pretrained DenseNet121
model = densenet121(pretrained=True)

# Modify the first conv layer to accept 4-channel input
model.features.conv0 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)

# Modify final layer for 9 classes
num_ftrs = model.classifier.in_features
model.classifier = nn.Linear(num_ftrs, 9)

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
