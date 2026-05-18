# ============================================
# Commit #3 - Visualizations
# Credit Card Fraud Detection
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
df = pd.read_csv("data/creditcard.csv")

fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

# ── Chart 1: Fraud vs Normal Count ─────────
plt.figure(figsize=(6, 4))
sns.countplot(x="Class", data=df, palette=["steelblue", "red"])
plt.title("Fraud vs Normal Transactions")
plt.xticks([0, 1], ["Normal", "Fraud"])
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("notebooks/chart1_class_distribution.png")
plt.show()
print("Chart 1 saved!")

# ── Chart 2: Transaction Amount ─────────────
plt.figure(figsize=(8, 4))
plt.hist(normal["Amount"], bins=50, alpha=0.6, label="Normal", color="steelblue")
plt.hist(fraud["Amount"], bins=50, alpha=0.6, label="Fraud", color="red")
plt.title("Transaction Amount Distribution")
plt.xlabel("Amount")
plt.ylabel("Count")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/chart2_amount_distribution.png")
plt.show()
print("Chart 2 saved!")

# ── Chart 3: Time Pattern ───────────────────
plt.figure(figsize=(10, 4))
plt.plot(normal["Time"], normal["Amount"], ".", alpha=0.1, color="steelblue", label="Normal")
plt.plot(fraud["Time"], fraud["Amount"], ".", alpha=0.5, color="red", label="Fraud")
plt.title("Transaction Time vs Amount")
plt.xlabel("Time")
plt.ylabel("Amount")
plt.legend()
plt.tight_layout()
plt.savefig("notebooks/chart3_time_pattern.png")
plt.show()
print("Chart 3 saved!")

# ── Chart 4: Correlation Heatmap ────────────
plt.figure(figsize=(14, 10))
sns.heatmap(df.corr(), cmap="coolwarm", center=0, linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("notebooks/chart4_correlation_heatmap.png")
plt.show()
print("Chart 4 saved!")

print("\nAll charts saved in notebooks/ folder!")