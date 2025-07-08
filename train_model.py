print("train_model.py started")

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib

# Paths
BASE_DATA_DIR = "./data"
IMAGES_DIR = os.path.join(BASE_DATA_DIR, "images")
train_csv_path = os.path.join(BASE_DATA_DIR, "train.csv")
test_csv_path = os.path.join(BASE_DATA_DIR, "test.csv")
MODEL_SAVE_PATH = "./models/random_forest_model.joblib"
SCALER_SAVE_PATH = "./models/feature_scaler.joblib"

target_cols = ['healthy', 'multiple_diseases', 'rust', 'scab']
IMG_HEIGHT = 128
IMG_WIDTH = 128
CHANNELS = 3

def extract_color_histogram_features(image_path, bins=32):
    image = cv2.imread(image_path)
    if image is None:
        return np.zeros(bins * CHANNELS)
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    hist_features = []
    for i in range(CHANNELS):
        hist = cv2.calcHist([image], [i], None, [bins], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        hist_features.extend(hist)
    return np.array(hist_features)

# Load data
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)

# Feature extraction
X_train_features = []
y_train_labels = []
for _, row in train_df.iterrows():
    image_id = row['image_id']
    image_full_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
    X_train_features.append(extract_color_histogram_features(image_full_path))
    y_train_labels.append(row[target_cols].values.astype(float))
X_train_features = np.array(X_train_features)
y_train_labels = np.array(y_train_labels)

# Train/val split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train_features, y_train_labels, test_size=0.2, random_state=42
)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)

# Model training
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, verbose=1)
model.fit(X_train_scaled, y_train_split)

# Validation predictions
y_val_pred_proba = model.predict_proba(X_val_scaled)
y_val_pred_proba_reshaped = np.array([proba[:, 1] for proba in y_val_pred_proba]).T

# Calculate mean column-wise ROC AUC
val_auc_score = roc_auc_score(y_val_split, y_val_pred_proba_reshaped, average='macro')
print(f"Validation Mean Column-wise ROC AUC: {val_auc_score:.4f}")

# Accuracy (exact match accuracy, where all labels must be correct)
threshold = 0.5
y_val_pred_binary = (y_val_pred_proba_reshaped > threshold).astype(int)
val_accuracy = accuracy_score(y_val_split, y_val_pred_binary)
print(f"Validation Accuracy (exact match, threshold {threshold}): {val_accuracy:.4f}")

# Save model and scaler
joblib.dump(model, MODEL_SAVE_PATH)
joblib.dump(scaler, SCALER_SAVE_PATH)
print("Model and scaler saved!")

print("Script finished successfully")