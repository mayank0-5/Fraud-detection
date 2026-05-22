import numpy as np
import joblib
from xgboost import XGBClassifier
X_train = np.load("data/X_train.npy")
y_train = np.load("data/y_train.npy")
print("Data loaded!")
print("Training XGBoost...")
model = XGBClassifier(eval_metric="logloss", random_state=42)
model.fit(X_train, y_train)
print("Training done!")
joblib.dump(model, "model.pkl")
print("Model saved as model.pkl!")
