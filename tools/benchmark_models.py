import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

# --- VRAM SAFEGUARD ---
# Prevent TensorFlow from hogging 100% of the RTX 5060 Ti VRAM so PyTorch (SB3) can breathe
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import keras
from keras import layers, models
from stable_baselines3 import PPO

# ==========================================
# 1. DATA PARSING & BACKGROUND SUBTRACTION
# ==========================================
def parse_csi_with_tare(data_csv, calib_csv, label):
    # --- STEP 1: CALCULATE THE ROOM'S BASELINE ---
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
    room_baseline = np.mean(calib_matrix, axis=0) 
    
    # --- STEP 2: SUBTRACT BASELINE FROM DYNAMIC DATA ---
    clean_rows = []
    with open(data_csv, 'r') as f:
        for line in f:
            match = re.search(r'\[(.*?)\]', line)
            if match:
                try:
                    nums = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
                    if len(nums) >= 128:
                        mags = [((nums[i]**2) + (nums[i+1]**2))**0.5 for i in range(0, 128, 2)]
                        
                        # Background subtraction (Tare weight)
                        subtracted_mags = np.abs(np.array(mags) - room_baseline)
                        clean_rows.append(subtracted_mags.tolist())
                except ValueError:
                    continue
                    
    df = pd.DataFrame(clean_rows)
    df['label'] = label
    return df

print("Parsing datasets and calculating background subtraction...")
# Label 0 = Authorized User, Label 1 = Intruder / Synthetic Anomaly
df_abid = parse_csi_with_tare('abid_data.csv', 'abid_calib.csv', 0)
df_stranger = parse_csi_with_tare('stranger_data.csv', 'stranger_calib.csv', 1)

data = pd.concat([df_abid, df_stranger], ignore_index=True)
X_raw = data.drop(columns=['label']).values.astype(np.float32)
y_raw = data['label'].values

# Normalize features globally between 0 and 1
if np.max(X_raw) > 0:
    X_raw = X_raw / np.max(X_raw)

# Create Sliding Windows (Window Size = 10 frames)
window_size = 10
X_windows, y_windows = [], []
for i in range(len(X_raw) - window_size):
    X_windows.append(X_raw[i : i + window_size])
    y_windows.append(y_raw[i + window_size - 1])

X = np.array(X_windows)
y = np.array(y_windows)

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# MODEL A: DEEP AUTOENCODER (Anomaly Score)
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
autoencoder.fit(X_train_safe, X_train_safe, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

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
cnn.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.1, verbose=1)

cnn_scores = cnn.predict(X_test).flatten()

# ==========================================
# MODEL C: PPO REINFORCEMENT LEARNING baseline
# ==========================================
print("\n--- Training PPO Reinforcement Learning Agent ---")
class BenchmarkRadarEnv(gym.Env):
    def __init__(self, X_data, y_data):
        super().__init__()
        self.X = X_data
        self.y = y_data
        self.idx = 0
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(10, 64), dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.idx = 0
        return self.X[self.idx], {}

    def step(self, action):
        true_label = self.y[self.idx]
        reward = 1 if action == true_label else -1
        self.idx += 1
        done = self.idx >= len(self.X)
        obs = self.X[self.idx] if not done else np.zeros((10, 64), dtype=np.float32)
        return obs, reward, done, False, {}

train_env = BenchmarkRadarEnv(X_train, y_train)
ppo_model = PPO("MlpPolicy", train_env, verbose=0, learning_rate=0.0005)
ppo_model.learn(total_timesteps=30000)

rl_scores = []
for obs in X_test:
    action, _states = ppo_model.predict(obs, deterministic=True)
    rl_scores.append(action)
rl_scores = np.array(rl_scores)

# ==========================================
# PLOTTING AND PORTFOLIO GENERATION
# ==========================================
print("\n--- Generating Metric Evaluations ---")
plt.figure(figsize=(10, 8))

fpr_ae, tpr_ae, _ = roc_curve(y_test, ae_scores)
plt.plot(fpr_ae, tpr_ae, label=f'Autoencoder (AUC = {auc(fpr_ae, tpr_ae):.3f})', color='blue')

fpr_cnn, tpr_cnn, _ = roc_curve(y_test, cnn_scores)
plt.plot(fpr_cnn, tpr_cnn, label=f'1D CNN Classifier (AUC = {auc(fpr_cnn, tpr_cnn):.3f})', color='green')

fpr_rl, tpr_rl, _ = roc_curve(y_test, rl_scores)
plt.plot(fpr_rl, tpr_rl, label=f'PPO RL Baseline (AUC = {auc(fpr_rl, tpr_rl):.3f})', color='red', linestyle='--')

plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing Baseline')
plt.xlabel('False Positive Rate (Stranger Accepted)')
plt.ylabel('True Positive Rate (Correct Alarm)')
plt.title('Wi-Fi CSI Biometric Bake-off: Tare Subtraction Active')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig('model_bakeoff_results.png')
print("\nSuccess! Results written out to 'model_bakeoff_results.png'.")