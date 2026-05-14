import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib
matplotlib.use('Agg')   

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras

# =========================
# 1. LOAD DATA
# =========================
print("\n🔹 STEP 1: DATA LOADING")

DATA_PATH = "data/processed/dataset.csv"
df = pd.read_csv(DATA_PATH)

print("✔ Dataset Loaded Successfully")
print("Shape:", df.shape)

# =========================
# 2. REMOVE IRRELEVANT COLUMN
# =========================
print("\n🔹 STEP 2: DROPPING IRRELEVANT FEATURES")

if "subject_id" in df.columns:
    df = df.drop("subject_id", axis=1)
    print("✔ Removed: subject_id")

# =========================
# 3. FEATURES & TARGET
# =========================
print("\n🔹 STEP 3: FEATURE & TARGET")

X = df.drop("stress_label", axis=1)
y = df["stress_label"]

print("Features:", list(X.columns))
print("Target: stress_label")

os.makedirs("models", exist_ok=True)
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

# =========================
# 4. TRAIN TEST SPLIT
# =========================
print("\n🔹 STEP 4: TRAIN-TEST SPLIT")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size :", X_test.shape[0])

# =========================
# 5. SCALING
# =========================
print("\n🔹 STEP 5: SCALING")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")
print("✔ Scaling Done")

# =========================
# 6. EDA
# =========================
print("\n🔹 STEP 6: EDA")

os.makedirs("outputs", exist_ok=True)

plt.figure()
sns.countplot(x=y)
plt.title("Stress Distribution")
plt.savefig("outputs/class_distribution.png")
plt.close()

plt.figure(figsize=(10,6))
sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/correlation.png")
plt.close()

print("✔ EDA Plots Saved")

# =========================
# 7. TUNED RANDOM FOREST
# =========================
print("\n🔹 STEP 7: TUNED RANDOM FOREST")

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

rf_train_acc = rf.score(X_train, y_train)
rf_test_acc = rf.score(X_test, y_test)

rf_pred = rf.predict(X_test)

print("\n===== TUNED RANDOM FOREST =====")
print("Train Accuracy :", round(rf_train_acc, 4))
print("Test Accuracy  :", round(rf_test_acc, 4))
print(classification_report(y_test, rf_pred))

joblib.dump(rf, "models/random_forest.pkl")

# Confusion Matrix
cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d")
plt.title("RF Confusion Matrix")
plt.savefig("outputs/rf_confusion_matrix.png")
plt.close()

# Feature Importance
importances = rf.feature_importances_
plt.figure(figsize=(8,5))
sns.barplot(x=importances, y=X.columns)
plt.title("Feature Importance (RF)")
plt.savefig("outputs/feature_importance.png")
plt.close()

# =========================
# 8. DEEP LEARNING
# =========================
print("\n🔹 STEP 8: DEEP LEARNING")

model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    X_train_scaled, y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=25,
    batch_size=32,
    verbose=1
)

dl_train_acc = history.history['accuracy'][-1]
dl_test_acc = history.history['val_accuracy'][-1]

dl_pred = np.argmax(model.predict(X_test_scaled), axis=1)

print("\n===== DEEP LEARNING =====")
print("Train Accuracy :", round(dl_train_acc, 4))
print("Test Accuracy  :", round(dl_test_acc, 4))
print(classification_report(y_test, dl_pred))

model.save("models/dl_model.keras")

# =========================
# 9. ENSEMBLE
# =========================
print("\n🔹 STEP 9: ENSEMBLE")

rf_prob = rf.predict_proba(X_test)
dl_prob = model.predict(X_test_scaled)

ensemble_pred = np.argmax((rf_prob + dl_prob)/2, axis=1)
ensemble_acc = accuracy_score(y_test, ensemble_pred)

print("\n===== ENSEMBLE =====")
print("Test Accuracy :", round(ensemble_acc, 4))
print(classification_report(y_test, ensemble_pred))

# =========================
# 10. FINAL SUMMARY
# =========================
print("\n🔹 FINAL SUMMARY")

print(f"RF Train Accuracy : {rf_train_acc:.4f}")
print(f"RF Test Accuracy  : {rf_test_acc:.4f}")
print(f"DL Train Accuracy : {dl_train_acc:.4f}")
print(f"DL Test Accuracy  : {dl_test_acc:.4f}")
print(f"Ensemble Accuracy : {ensemble_acc:.4f}")

# =========================
# 11. MODEL ACCURACY EDA
# =========================
print("\n🔹 STEP 11: MODEL ACCURACY EDA")

models = ["Random Forest", "Deep Learning", "Ensemble"]
accuracies = [rf_test_acc, dl_test_acc, ensemble_acc]

plt.figure(figsize=(7,5))
sns.barplot(x=models, y=accuracies)
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.9, 1.0)

for i, v in enumerate(accuracies):
    plt.text(i, v + 0.002, f"{v:.3f}", ha='center')

plt.savefig("outputs/model_accuracy_comparison.png")
plt.close()

print("✔ Saved: outputs/model_accuracy_comparison.png")

# DL Curve
plt.figure(figsize=(7,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Deep Learning Accuracy Curve")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.savefig("outputs/dl_accuracy_curve.png")
plt.close()

print("✔ Saved: outputs/dl_accuracy_curve.png")

print("\n✅ PIPELINE COMPLETE")