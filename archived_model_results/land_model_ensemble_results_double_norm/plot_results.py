import matplotlib.pyplot as plt
import pandas as pd

# Manually enter your results
cv_results = {
    "Fold": [1, 2, 3, 4, 5],
    "R2": [0.4281, 0.7090, 0.4810, 0.3564, 0.6081],
    "RMSE": [86.4608, 63.0774, 54.0305, 87.1664, 64.1367],
    "MAE": [18.1628, 16.1526, 15.8115, 15.6994, 15.0832]
}

test_set = {
    "R2": 0.6313,
    "RMSE": 60.9881,
    "MAE": 15.9884
}

df = pd.DataFrame(cv_results)

# --- Plot 1: R² per fold and test ---
plt.figure(figsize=(10, 4))
plt.subplot(1, 3, 1)
plt.plot(df["Fold"], df["R2"], marker='o', label="CV Folds")
plt.axhline(test_set["R2"], color='red', linestyle='--', label="Test")
plt.ylim(0, 1)
plt.title("R² Score")
plt.xlabel("Fold")
plt.ylabel("R²")
plt.legend()

# --- Plot 2: RMSE ---
plt.subplot(1, 3, 2)
plt.plot(df["Fold"], df["RMSE"], marker='o', label="CV Folds")
plt.axhline(test_set["RMSE"], color='red', linestyle='--', label="Test")
plt.title("RMSE (mm)")
plt.xlabel("Fold")
plt.ylabel("RMSE")
plt.legend()

# --- Plot 3: MAE ---
plt.subplot(1, 3, 3)
plt.plot(df["Fold"], df["MAE"], marker='o', label="CV Folds")
plt.axhline(test_set["MAE"], color='red', linestyle='--', label="Test")
plt.title("MAE (mm)")
plt.xlabel("Fold")
plt.ylabel("MAE")
plt.legend()

plt.tight_layout()
plt.suptitle("Cross-Validation vs Test Performance", y=1.05)
plt.savefig("cv_test_results.png", dpi=300, bbox_inches='tight')
plt.show()
