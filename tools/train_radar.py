import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
import re

class RadarGuardEnv(gym.Env):
    def __init__(self, empty_csv, intruder_csv, window_size=10):
        super(RadarGuardEnv, self).__init__()
        
        # Parse the raw ESP-IDF serial lines safely to ignore text headers
        self.data = pd.concat([
            self._parse_csi(empty_csv, 0),
            self._parse_csi(intruder_csv, 1)
        ], ignore_index=True)
        
        self.csi_features = self.data.drop(columns=['label']).values.astype(np.float32)
        self.labels = self.data['label'].values
        
        self.window_size = window_size
        self.num_subcarriers = self.csi_features.shape[1]
        
        self.action_space = spaces.Discrete(2) # 0: Sleep, 1: Alarm
        self.observation_space = spaces.Box(
            low=0, high=255, 
            shape=(self.window_size, self.num_subcarriers), 
            dtype=np.float32
        )
        self.current_idx = self.window_size

    def _parse_csi(self, csv_file, label):
        clean_rows = []
        with open(csv_file, 'r') as f:
            for line in f:
                # Find the data array inside the brackets: [12, -4, 5, ...]
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    try:
                        # Extract the integer array
                        nums = [int(x.strip()) for x in match.group(1).split(',') if x.strip()]
                        
                        # 64 Wi-Fi subcarriers * 2 (I/Q pairs) = 128 numbers
                        if len(nums) >= 128:
                            # Calculate Wave Amplitude: sqrt(I^2 + Q^2)
                            mags = [((nums[i]**2) + (nums[i+1]**2))**0.5 for i in range(0, 128, 2)]
                            clean_rows.append(mags)
                    except ValueError:
                        continue
        df = pd.DataFrame(clean_rows)
        if df.empty:
            raise ValueError(f"No valid bracketed CSI data found in {csv_file}. Check your recorded files!")
        df['label'] = label
        return df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_idx = self.window_size
        return self._get_obs(), {}

    def _get_obs(self):
        return self.csi_features[self.current_idx - self.window_size : self.current_idx]

    def step(self, action):
        true_label = self.labels[self.current_idx]
        
        if action == 0:  # Agent chose Sleep
            reward = 1 if true_label == 0 else -50
        else:  # Agent chose Alarm
            reward = 10 if true_label == 1 else -10
                
        self.current_idx += 1
        done = self.current_idx >= len(self.data) - 1
        
        obs = self._get_obs() if not done else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, done, False, {}

if __name__ == "__main__":
    print("Parsing datasets...")
    env = RadarGuardEnv('empty_room.csv', 'intruder_movement.csv')
    print(f"Loaded {len(env.data)} valid CSI packets. Starting training...")
    
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=0.0005)
    model.learn(total_timesteps=100000)
    
    model.save("radar_guard_policy")
    print("Done! Saved as radar_guard_policy.zip")