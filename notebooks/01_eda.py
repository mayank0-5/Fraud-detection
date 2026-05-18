# ============================================
# Commit #2 - Data Loading & EDA
# Credit Card Fraud Detection
# ============================================

import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv("data/creditcard.csv")

# Basic Info
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nBasic Statistics:")
print(df.describe())

# Fraud vs Normal
fraud = df[df["Class"] == 1]
normal = df[df["Class"] == 0]

print(f"\nTotal Transactions : {len(df)}")
print(f"Normal Transactions: {len(normal)}")
print(f"Fraud Transactions : {len(fraud)}")
print(f"Fraud Percentage   : {len(fraud)/len(df)*100:.2f}%")