import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models

# Dataset path
DATASET_PATH = r"E:\RGB-NIR\nirscene1"
SAVE_PATH = r"E:\RGB-NIR"  # Path to store extracted features

# Define transformations for ResNet
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize for ResNet
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize using ResNet mean and std
])

# Custom dataset class
class RGB_Dataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.data = []

        for label, class_name in enumerate(self.classes):
            class_path = os.path.join(root_dir, class_name)
            rgb_images = [f for f in os.listdir(class_path) if f.endswith("_rgb.tiff")]

            for rgb_file in rgb_images:
                self.data.append((os.path.join(class_path, rgb_file), label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path, label = self.data[idx]
        rgb_image = cv2.imread(rgb_path, cv2.IMREAD_UNCHANGED)

        if self.transform:
            rgb_image = self.transform(rgb_image)

        return rgb_image, label

# Create dataset and dataloader
dataset = RGB_Dataset(DATASET_PATH, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load Pretrained ResNet without the classifier (to extract features)
model = models.resnet18(pretrained=True)  # You can choose resnet50, resnet34, etc.
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

# Remove the classifier (fully connected layer)
feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove the last layer (fc)
feature_extractor.eval()

# Extract and save features
all_features = []
all_labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)

        # Extract features
        features = feature_extractor(images)  # Extract features
        features = features.view(features.size(0), -1)  # Flatten the features

        all_features.append(features.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_features = np.vstack(all_features)
all_labels = np.array(all_labels)

# Save features and labels to disk
features_file = os.path.join(SAVE_PATH, "resnet_rgb_features.npy")
labels_file = os.path.join(SAVE_PATH, "labels.npy")

np.save(features_file, all_features)
np.save(labels_file, all_labels)

print(f"Feature extraction complete. Features saved to '{features_file}' and labels to '{labels_file}'.")

# Optionally print the shapes of the features and labels
print(all_features.shape, all_labels.shape)
