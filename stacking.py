import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

# Load the pre-extracted features
nir_features = np.load(r"E:\RGB-NIR\efficientnet_nir_features.npy")
rgb_features = np.load(r"E:\RGB-NIR\efficientnet_rgb_features.npy")
labels = np.load(r"E:\RGB-NIR\labels.npy")

# Concatenate NIR and RGB features
features = np.concatenate((nir_features, rgb_features), axis=1)

# Split the data into training and testing (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define 5 Base Models (Popular ML Algorithms)
base_models = {
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "XGBoost": xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss"),
    "NaiveBayes": GaussianNB(),
}

# Train Base Models and Collect Predictions
train_meta_features = np.zeros((X_train.shape[0], len(base_models)))  # Store base model outputs for training
test_meta_features = np.zeros((X_test.shape[0], len(base_models)))  # Store base model outputs for testing

for i, (name, model) in enumerate(base_models.items()):
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Get predictions (probabilities if supported, else class labels)
    if hasattr(model, "predict_proba"):
        train_meta_features[:, i] = np.argmax(model.predict_proba(X_train), axis=1)
        test_meta_features[:, i] = np.argmax(model.predict_proba(X_test), axis=1)
    else:
        train_meta_features[:, i] = model.predict(X_train)
        test_meta_features[:, i] = model.predict(X_test)

# Train Meta-Model (Logistic Regression)
meta_model = LogisticRegression(max_iter=1000)
meta_model.fit(train_meta_features, y_train)

# Predict using Stacked Model
y_pred = meta_model.predict(test_meta_features)

# Evaluate Final Model
accuracy = accuracy_score(y_test, y_pred)
print(f"\nFinal Stacked Model Test Accuracy: {accuracy * 100:.2f}%")
