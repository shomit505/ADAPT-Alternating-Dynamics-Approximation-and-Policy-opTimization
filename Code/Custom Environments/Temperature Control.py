import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

class TemperatureControlEnv(gym.Env):
    def __init__(self):
        super(TemperatureControlEnv, self).__init__()

        self.action_space = spaces.Discrete(2)  # 0, 1
        self.observation_space = spaces.Box(low=15.0, high=30.0, shape=(1,), dtype=np.float32)

        self.state = None
        self.target_temp = 22.5
        self.threshold = 0.5
        self.steps = 0
        self.max_steps = 100

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = random.uniform(15.0, 30.0)
        self.steps = 0
        return np.array([self.state]), {}

    def step(self, action):
        self.steps += 1
        if action == 0:
          self.state += -0.25 # move left
        if action == 1:
          self.state += 0.25 # move right
        #+ self.np_random.normal(0, 0.1, size=(1,))
        self.state = np.clip(self.state, self.observation_space.low, self.observation_space.high).astype(np.float32)

        temp_diff = np.abs(self.state - self.target_temp).item()
        if temp_diff <= self.threshold:
            reward = 0
        else:
            reward = -1
        terminated = temp_diff <= self.threshold
        truncated = self.steps >= self.max_steps

        return self.state, reward, terminated, truncated, {}

    def render(self):
        print(f"Current temperature: {self.state[0]:.2f}°C, Target: {self.target_temp:.2f}°C, Steps: {self.steps}")

gym.register(
    id='TemperatureControl-v0',
    entry_point='__main__:TemperatureControlEnv',
    max_episode_steps=100,
)