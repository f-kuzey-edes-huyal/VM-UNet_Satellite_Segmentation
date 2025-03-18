import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Data
metrics = ["Loss", "mIoU", "F1/DSC", "Accuracy", "Specificity", "Sensitivity"]
no_aug = [0.1854, 0.6917, 0.8178, 0.7683, 0.7439, 0.7805]
aug = [0.1842, 0.6949, 0.8199, 0.7743, 0.7794, 0.7717]

# Set plot style
sns.set_style("whitegrid")

# Bar plot
x = np.arange(len(metrics))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width/2, no_aug, width, label="No Augmentation", color="gray", alpha=0.7)
bars2 = ax.bar(x + width/2, aug, width, label="With Augmentation", color="cornflowerblue", alpha=0.7)

# Labels
ax.set_xlabel("Metrics", fontsize=12)
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Performance Comparison (With vs Without Augmentations)", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=25, ha="right")
ax.legend()

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

# Save the figure
plt.tight_layout()
plt.savefig("performance_comparison.png", dpi=300, bbox_inches="tight")

# Show the plot
plt.show()
