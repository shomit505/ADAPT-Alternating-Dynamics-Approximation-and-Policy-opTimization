class DeepSeaEnv(gym.Env):
    def __init__(self, N=5):
        super(DeepSeaEnv, self).__init__()
        self.N = N
        self.action_space = spaces.Discrete(2)  # 0: Left, 1: Right
        self.observation_space = spaces.Discrete(N * N)  # Flattened grid
        self.penalty = -0.01 / N
        self.final_reward = 1.0
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = (0, 0)  # Start at the top-left corner
        self.steps_taken = 0
        return self._get_obs(), {}

    def _get_obs(self):
        row, col = self.state
        return row * self.N + col

    def step(self, action):
        row, col = self.state

        # Determine new column based on action
        if action == 1:  # Right
            new_col = min(col + 1, self.N - 1)
            reward = self.penalty
        else:  # Left
            new_col = max(col - 1, 0)
            reward = 0

        # Move down one row
        new_row = min(row + 1, self.N - 1)  # Ensure we don't go beyond the bottom row

        self.state = (new_row, new_col)
        self.steps_taken += 1

        # Check if we have reached the terminal state
        terminated = new_row == self.N - 1  # Terminate when we reach the bottom row
        truncated = False

        # If at the bottom-right corner
        if terminated and new_col == self.N - 1:
            reward += self.final_reward

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        grid = np.zeros((self.N, self.N))
        row, col = self.state
        grid[row, col] = 1
        print(grid)

    def close(self):
        pass

# Register the environment
gym.register(
    id='DeepSea-v0',
    entry_point=DeepSeaEnv,
    kwargs={'N': 14},
)

# Create the environment
env = gym.make('DeepSea-v0')