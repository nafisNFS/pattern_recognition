import matplotlib.pyplot as plt
import numpy as np

# Accuracy data
models = [
    "ResNet (RGB-NIR Stacked)",
    "EfficientNet (RGB-NIR Stacked)",
    "ResNet (Only RGB)",
    "EfficientNet (Only RGB)",
    "ResNet (Only NIR)",
    "EfficientNet (Only NIR)",
    "Proposed Dual Channel CNN"
]
accuracies = [72.00, 73.00, 77.00, 84.00, 75.00, 75.00, 90.56]

# Bar chart setup
plt.figure(figsize=(10, 6))
x = np.arange(len(models))

# Plot bars with striped pattern similar to the reference
bars = plt.bar(x, accuracies, color='skyblue', edgecolor='black', hatch='//')

# Add data labels
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', fontsize=10)

# Formatting
plt.xticks(x, models, rotation=45, ha='right', fontsize=10)
plt.ylabel("Accuracy (%)", fontsize=12)
plt.title("Comparison of Model Accuracies", fontsize=14)
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show plot
plt.tight_layout()
plt.show()
