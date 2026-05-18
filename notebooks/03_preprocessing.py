# ============================================
# Commit #4 - Preprocessing + SMOTE
# Credit Card Fraud Detection
# ============================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# ── Load Data ──────────────────────────────
df = pd.read_csv("data/creditcard.csv")
print("Dataset loaded! Shape:", df.shape)

# ── Step 1: Scale Amount and Time ──────────
scaler = StandardScaler()
df["Amount_Scaled"] = scaler.fit_transform(df[["Amount"]])
df["Time_Scaled"] = scaler.fit_transform(df[["Time"]])

# Drop original columns
df = df.drop(["Amount", "Time"], axis=1)
print("\nScaling done!")

# ── Step 2: Split Features and Target ──────
X = df.drop("Class", axis=1)
y = df["Class"]

print(f"\nFeatures shape : {X.shape}")
print(f"Target shape   : {y.shape}")
print(f"\nBefore SMOTE:")
print(f"Normal : {sum(y == 0)}")
print(f"Fraud  : {sum(y == 1)}")

# ── Step 3: Train Test Split ────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape}")
print(f"Test size : {X_test.shape}")

# ── Step 4: Apply SMOTE ─────────────────────
smote = SMOTE(random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Normal : {sum(y_train_sm == 0)}")
print(f"Fraud  : {sum(y_train_sm == 1)}")
print("\nData is now balanced and ready for training!")

# ── Step 5: Save Processed Data ────────────
np.save("data/X_train.npy", X_train_sm)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train_sm)
np.save("data/y_test.npy", y_test)
print("\nProcessed data saved in data/ folder!")