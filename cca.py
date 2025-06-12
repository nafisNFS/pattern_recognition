import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA

# Load the pre-extracted features
nir_features = np.load(r"E:\RGB-NIR\efficientnet_nir_features.npy")
rgb_features = np.load(r"E:\RGB-NIR\efficientnet_rgb_features.npy")
labels = np.load(r"E:\RGB-NIR\labels.npy")

pca = PCA(n_components=100)  # Reduce to 100 dimensions first
nir_pca = pca.fit_transform(nir_features)
rgb_pca = pca.fit_transform(rgb_features)

cca = CCA(n_components=50)  # Now apply CCA
nir_transformed, rgb_transformed = cca.fit_transform(nir_pca, rgb_pca)

nir_transformed, rgb_transformed = cca.fit_transform(nir_features, rgb_features)

# Concatenate the transformed features
cca_features = np.concatenate((nir_transformed, rgb_transformed), axis=1)

# Split the data into training and testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(cca_features, labels, test_size=0.2, random_state=42)

# Convert the data to torch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)


# Create a custom dataset class
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


# Create DataLoaders for training and testing
train_dataset = FeatureDataset(X_train_tensor, y_train_tensor)
test_dataset = FeatureDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)


# Define a simple classifier based on EfficientNet's fully connected layer
class EfficientNetClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(EfficientNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)


# Initialize the classifier
input_dim = X_train.shape[1]  # Input dimension from CCA-transformed features
num_classes = len(np.unique(labels))  # Number of unique classes in the dataset
model = EfficientNetClassifier(input_dim, num_classes)

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

epochs = 10  # Define number of epochs

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for features, labels in train_loader:
        features, labels = features.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Print statistics
    print(
        f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

# Evaluate the model
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for features, labels in test_loader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features)
        _, predicted = torch.max(outputs, 1)

        y_pred.extend(predicted.cpu().numpy())
        y_true.extend(labels.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
