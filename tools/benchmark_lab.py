import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from keras import layers, models

# VRAM SAFEGUARD
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        pass

def parse_csi_with_tare(data_csv, calib_csv, label):
    # STEP 1: CALCULATE THE ROOM'S BASELINE
    calib_rows = []
    with open(calib_csv, 'r') as f:
        for line in f:
            match = re.search(r'\[(.*?)\]', line)
            if match:
                try:
                    nums = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
                    if len(nums) >= 128:
                        mags = [((nums[i]**2) + (nums[i+1]**2))**0.5 for i in range(0, 128, 2)]
                        calib_rows.append(mags)
                except ValueError:
                    continue
                    
    calib_matrix = np.array(calib_rows)
    if len(calib_matrix) == 0:
        return pd.DataFrame()
    room_baseline = np.mean(calib_matrix, axis=0) 
    
    # STEP 2: SUBTRACT BASELINE FROM DYNAMIC DATA
    clean_rows = []
    with open(data_csv, 'r') as f:
        for line in f:
            match = re.search(r'\[(.*?)\]', line)
            if match:
                try:
                    nums = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
                    if len(nums) >= 128:
                        mags = [((nums[i]**2) + (nums[i+1]**2))**0.5 for i in range(0, 128, 2)]
                        subtracted_mags = np.abs(np.array(mags) - room_baseline)
                        clean_rows.append(subtracted_mags.tolist())
                except ValueError:
                    continue
                    
    df = pd.DataFrame(clean_rows)
    if not df.empty:
        df['label'] = label
    return df

print("Parsing multi-target lab datasets...")
# Label 0 = Authorized User (Abid)
df_abid = parse_csi_with_tare('abid_data.csv', 'abid_calib.csv', 0)

# Label 1 = Intruders / Unauthorized
df_nowreen = parse_csi_with_tare('nowreen_data.csv', 'nowreen_calib.csv', 1)
df_sakib = parse_csi_with_tare('sakib_data.csv', 'sakib_calib.csv', 1)
df_sayma = parse_csi_with_tare('sayma_data.csv', 'sayma_calib.csv', 1)

# Combine all data
data = pd.concat([df_abid, df_nowreen, df_sakib, df_sayma], ignore_index=True)
X_raw = data.drop(columns=['label']).values.astype(np.float32)
y_raw = data['label'].values

# Normalize
if np.max(X_raw) > 0:
    X_raw = X_raw / np.max(X_raw)

# Sliding Windows (10 frames)
window_size = 10
X_windows, y_windows = [], []
for i in range(len(X_raw) - window_size):
    X_windows.append(X_raw[i : i + window_size])
    y_windows.append(y_raw[i + window_size - 1])

X = np.array(X_windows)
y = np.array(y_windows)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# MODEL A: DEEP AUTOENCODER
# ==========================================
print("\n--- Training Autoencoder (Baseline Profile) ---")
X_train_safe = X_train[y_train == 0]

autoencoder = models.Sequential([
    layers.Input(shape=(10, 64)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(640, activation='sigmoid'),
    layers.Reshape((10, 64))
])

autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X_train_safe, X_train_safe, epochs=15, batch_size=32, validation_split=0.1, verbose=1)

reconstructions = autoencoder.predict(X_test)
ae_scores = np.mean(np.square(X_test - reconstructions), axis=(1, 2))

# ==========================================
# MODEL B: SUPERVISED 1D CNN
# ==========================================
print("\n--- Training 1D CNN Classifier ---")
cnn = models.Sequential([
    layers.Input(shape=(10, 64)),
    layers.Conv1D(32, kernel_size=3, activation='relu'),
    layers.MaxPooling1D(2),
    layers.Flatten(),
    layers.Dense(32, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

cnn.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=1)

cnn_scores = cnn.predict(X_test).flatten()

# ==========================================
# PLOTTING
# ==========================================
print("\n--- Generating Metric Evaluations ---")
plt.figure(figsize=(10, 8))

fpr_ae, tpr_ae, _ = roc_curve(y_test, ae_scores)
plt.plot(fpr_ae, tpr_ae, label=f'Autoencoder (AUC = {auc(fpr_ae, tpr_ae):.3f})', color='blue', linewidth=2)

fpr_cnn, tpr_cnn, _ = roc_curve(y_test, cnn_scores)
plt.plot(fpr_cnn, tpr_cnn, label=f'1D CNN Classifier (AUC = {auc(fpr_cnn, tpr_cnn):.3f})', color='green', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing Baseline')
plt.xlabel('False Positive Rate (Intruder Accepted)')
plt.ylabel('True Positive Rate (Correct Alarm)')
plt.title('Multi-Target Biometric Bake-off: Real Lab Data')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('real_lab_results.png')
print("\nSuccess! Results written out to 'real_lab_results.png'.")