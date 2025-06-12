import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torch import nn, optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report

# Load the pre-extracted features from ResNet & EfficientNet (NIR + RGB)
resnet_nir = np.load(r"E:\RGB-NIR\resnet_nir_features.npy")
resnet_rgb = np.load(r"E:\RGB-NIR\resnet_rgb_features.npy")
efficientnet_nir = np.load(r"E:\RGB-NIR\efficientnet_nir_features.npy")
efficientnet_rgb = np.load(r"E:\RGB-NIR\efficientnet_rgb_features.npy")

# Load labels
labels = np.load(r"E:\RGB-NIR\labels.npy")

# Concatenate all feature vectors
all_features = np.concatenate((resnet_nir, resnet_rgb, efficientnet_nir, efficientnet_rgb), axis=1)

# Convert to PyTorch tensors
X_tensor = torch.tensor(all_features, dtype=torch.float32)
y_tensor = torch.tensor(labels, dtype=torch.long)

# Custom dataset class
class FeatureDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Define classifier model
class MultiFeatureClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(MultiFeatureClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)

# Initialize KFold
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = X_tensor.shape[1]
num_classes = len(np.unique(labels))
epochs = 10
batch_size = 16

# Store results for each fold
fold_results = {
    'train_losses': [],
    'train_accuracies': [],
    'test_accuracies': [],
    'precisions': [],
    'recalls': [],
    'f1_scores': [],
    'classwise_metrics': []  # To store class-wise metrics for each fold
}

# 5-Fold Cross-Validation
for fold, (train_idx, test_idx) in enumerate(kfold.split(X_tensor)):
    print(f"\nFold {fold + 1}")

    # Split data into train and test for this fold
    X_train, X_test = X_tensor[train_idx], X_tensor[test_idx]
    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]

    # Create DataLoaders
    train_dataset = FeatureDataset(X_train, y_train)
    test_dataset = FeatureDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, and optimizer
    model = MultiFeatureClassifier(input_dim, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_losses = []
    train_accuracies = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {train_loss:.4f}, Accuracy: {train_acc:.4f}%")

    # Store training results for this fold
    fold_results['train_losses'].append(train_losses)
    fold_results['train_accuracies'].append(train_accuracies)

    # Evaluation on test set
    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(labels.cpu().numpy())

    # Calculate overall metrics for this fold
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

    # Store overall test results for this fold
    fold_results['test_accuracies'].append(accuracy)
    fold_results['precisions'].append(precision)
    fold_results['recalls'].append(recall)
    fold_results['f1_scores'].append(f1)

    # Calculate class-wise metrics for this fold
    classwise_report = classification_report(y_true, y_pred, output_dict=True, digits=4)
    fold_results['classwise_metrics'].append(classwise_report)

    print(f"\nTest Accuracy (Fold {fold + 1}): {accuracy * 100:.4f}%")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")
    print("\nClass-wise Metrics (Fold {fold + 1}):")
    print(classification_report(y_true, y_pred, digits=4))

# Print average results across all folds
print("\nAverage Results Across 5 Folds:")
print(f"Test Accuracy: {np.mean(fold_results['test_accuracies']) * 100:.4f}%")
print(f"Precision: {np.mean(fold_results['precisions']):.4f}")
print(f"Recall: {np.mean(fold_results['recalls']):.4f}")
print(f"F1 Score: {np.mean(fold_results['f1_scores']):.4f}")

# Print average class-wise metrics across all folds
print("\nAverage Class-wise Metrics Across 5 Folds:")
classwise_metrics_avg = {}
for class_name in fold_results['classwise_metrics'][0].keys():
    if class_name.isdigit():  # Only process class labels (skip 'accuracy', 'macro avg', etc.)
        classwise_metrics_avg[class_name] = {
            'precision': np.mean([fold_results['classwise_metrics'][i][class_name]['precision'] for i in range(5)]),
            'recall': np.mean([fold_results['classwise_metrics'][i][class_name]['recall'] for i in range(5)]),
            'f1-score': np.mean([fold_results['classwise_metrics'][i][class_name]['f1-score'] for i in range(5)]),
            'support': fold_results['classwise_metrics'][0][class_name]['support']  # Support is the same across folds
        }

# Print class-wise metrics
for class_name, metrics in classwise_metrics_avg.items():
    print(f"\nClass {class_name}:")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1-score']:.4f}")
    print(f"Support: {metrics['support']}")

# Plot Training Accuracy and Loss for the last fold
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs + 1), train_accuracies, label='Training Accuracy', marker='o', color='blue')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training Accuracy Over Epochs (Last Fold)')
plt.legend()
plt.grid()

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs + 1), train_losses, label='Training Loss', marker='s', color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs (Last Fold)')
plt.legend()
plt.grid()

plt.show()