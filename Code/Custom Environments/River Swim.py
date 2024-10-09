class RiverSwimEnv(gym.Env):
    def __init__(self, nS=6):
        super(RiverSwimEnv, self).__init__()
        self.nS = nS
        self.nA = 2  # LEFT = 0, RIGHT = 1
        self.state = 0
        self.steps_taken = 0
        self.max_steps = 20

        self.action_space = spaces.Discrete(self.nA)
        self.observation_space = spaces.Discrete(self.nS)

        # Define transition probabilities and rewards
        self.P = self._init_dynamics()

    def _init_dynamics(self):
        P = {}
        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}

        # LEFT transitions
        for s in range(self.nS):
            P[s][0] = [(1.0, max(0, s-1), 5/1000 if s == 0 else 0, False)]

        # RIGHT transitions
        P[0][1] = [(0.3, 0, 0, False), (0.7, 1, 0, False)]
        for s in range(1, self.nS - 1):
            P[s][1] = [
                (0.1, max(0, s-1), 0, False),
                (0.6, s, 0, False),
                (0.3, min(self.nS-1, s+1), 0, False)
            ]
        P[self.nS-1][1] = [(0.7, self.nS-1, 1, False), (0.3, self.nS-2, 0, False)]

        return P

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = 0
        self.steps_taken = 0
        return self.state, {}

    def step(self, action):
        transitions = self.P[self.state][action]
        i = self.np_random.choice(len(transitions), p=[t[0] for t in transitions])
        p, next_state, reward, _ = transitions[i]
        self.state = next_state
        self.steps_taken += 1

        # Check if max steps reached
        done = self.steps_taken >= self.max_steps

        return next_state, reward, done, False, {}

    def render(self):
        print(f"Current state: {self.state}")

# Register the environment
gym.register(
    id='RiverSwim-v0',
    entry_point='__main__:RiverSwimEnv',
    max_episode_steps=20,
)

# Create the environment
env = gym.make('RiverSwim-v0')