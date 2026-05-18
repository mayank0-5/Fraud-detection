# ============================================
# Commit #5 - Model Training & Evaluation
# Credit Card Fraud Detection
# ============================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)

# ── Load Processed Data ─────────────────────
X_train = np.load("data/X_train.npy")
X_test  = np.load("data/X_test.npy")
y_train = np.load("data/y_train.npy")
y_test  = np.load("data/y_test.npy")
print("Data loaded successfully!")

# ── Define Models ───────────────────────────
models = {
    "Logistic Regression" : LogisticRegression(max_iter=1000),
    "Random Forest"       : RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost"             : XGBClassifier(eval_metric="logloss", random_state=42)
}

results = {}

# ── Train & Evaluate Each Model ─────────────
for name, model in models.items():
    print(f"\n Training {name}...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    auc    = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    print(f"✅ {name} done!")
    print(f"   ROC-AUC Score : {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred))

    results[name] = auc

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    disp.plot(cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"notebooks/confusion_matrix_{name.replace(' ', '_')}.png")
    plt.show()
    print(f"Confusion matrix saved!")

# ── Compare All Models ──────────────────────
print("\n" + "="*45)
print("         MODEL COMPARISON (ROC-AUC)")
print("="*45)
for name, auc in results.items():
    bar = "█" * int(auc * 40)
    print(f"{name:22} : {auc:.4f} {bar}")

best_model = max(results, key=results.get)
print(f"\n🏆 Best Model: {best_model} with AUC: {results[best_model]:.4f}")