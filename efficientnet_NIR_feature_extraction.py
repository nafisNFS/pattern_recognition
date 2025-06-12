import os
import cv2
import torch
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

# Dataset path
DATASET_PATH = r"E:\RGB-NIR\nirscene1"
SAVE_PATH = r"E:\RGB-NIR"  # Path to store extracted features

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224)),  # Resize for EfficientNet
    transforms.Lambda(lambda x: x.expand(3, -1, -1)),  # Convert single-channel to 3-channel
    transforms.Normalize([0.5] * 3, [0.5] * 3)  # Normalize 3 channels
])

# Custom dataset class for NIR images
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

        return nir_image, label

# Create dataset and dataloader
dataset = NIR_Dataset(DATASET_PATH, transform)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Load Pretrained EfficientNet without warnings
model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

# Remove classifier to extract features
feature_extractor = nn.Sequential(*list(model.children())[:-1])  # Remove last layer
feature_extractor = feature_extractor.to("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor.eval()

# Extract and save features
all_features = []
all_labels = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with torch.no_grad():
    for images, labels in dataloader:
        images = images.to(device)

        features = feature_extractor(images)  # Extract features
        features = features.view(features.size(0), -1)  # Flatten

        all_features.append(features.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_features = np.vstack(all_features)
all_labels = np.array(all_labels)

# Save features and labels to E:\RGB-NIR
features_file = os.path.join(SAVE_PATH, "efficientnet_nir_features.npy")
labels_file = os.path.join(SAVE_PATH, "labels.npy")

np.save(features_file, all_features)
np.save(labels_file, all_labels)

print(f"Feature extraction complete. Features saved to '{features_file}' and labels to '{labels_file}'.")

# Load the saved features and labels to verify
features = np.load(r"E:\RGB-NIR\efficientnet_nir_features.npy")
labels = np.load(r"E:\RGB-NIR\labels.npy")
print(features.shape, labels.shape)
