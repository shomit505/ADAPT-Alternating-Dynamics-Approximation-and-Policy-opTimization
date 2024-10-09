import torch
import torch.optim as optim
import numpy as np
import gym
from collections import deque

N = 14

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def normalize_slices(tensor):
    return tensor / tensor.sum(dim=-1, keepdim=True)

def normalise_initial(counts):
    return counts / counts.sum()

def softmax_policy(policy_table):
    return torch.nn.functional.softmax(policy_table, dim=-1)

def state_to_index(state, env):
    if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        # Calculate the index for MultiDiscrete space
        index = 0
        for i, (s, n) in enumerate(zip(state, env.observation_space.nvec)):
            index += s * np.prod(env.observation_space.nvec[i+1:])
        return int(index)
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        return state
    else:
        raise ValueError("Unsupported observation space type")

def get_num_states(env):
    if isinstance(env.observation_space, gym.spaces.MultiDiscrete):
        return np.prod(env.observation_space.nvec)
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        return env.observation_space.n
    elif hasattr(env.observation_space, 'n'):
        return env.observation_space.n
    else:
        raise ValueError(f"Unsupported observation space type: {type(env.observation_space)}")

def sample_steps(env, policy, num_steps, max_steps_per_episode):
    num_states = get_num_states(env)
    num_actions = env.action_space.n
    trajectories = []
    initial_states = []
    transition_counts = torch.ones((num_states, num_actions, num_states), dtype=torch.int32, device=device)
    reward_total = torch.zeros((num_states, num_actions), device=device)
    reward_count = torch.zeros((num_states, num_actions), device=device)
    initial_state_count = torch.zeros(num_states, dtype=torch.int32, device=device)

    steps_taken = 0
    while steps_taken < num_steps:
        state, _ = env.reset()
        state_idx = state
        initial_state_count[state_idx] += 1
        initial_states.append(state_idx)
        trajectory = []

        for step in range(max_steps_per_episode):
            action = torch.multinomial(policy[state_idx].cpu(), 1).item()
            next_state, reward, done, _, _ = env.step(action)
            next_state_idx = next_state
            trajectory.append((state_idx, action, reward, next_state_idx))

            transition_counts[state_idx, action, next_state_idx] += 1
            reward_total[state_idx, action] += reward
            reward_count[state_idx, action] += 1

            steps_taken += 1
            if done or steps_taken >= num_steps:
                break
            state_idx = next_state_idx

        trajectories.append(trajectory)

    return transition_counts, reward_total, reward_count, initial_state_count, initial_states, trajectories, steps_taken

def process_trajectories(trajectories):
    states = []
    actions = []
    rewards = []
    next_states = []

    for trajectory in trajectories:
        for step in trajectory:
            states.append(step[0])
            actions.append(step[1])
            rewards.append(step[2])
            next_states.append(step[3])

    return (torch.tensor(states, device=device),
            torch.tensor(actions, device=device),
            torch.tensor(rewards, dtype=torch.float32, device=device),
            torch.tensor(next_states, device=device))

def compute_J_counting(env, policy, v, R, P, gamma=0.99):
    num_states = P.shape[0]
    P_a = P.permute(1, 0, 2)
    P_pi = torch.einsum('sa,ask->sk', policy, P_a)
    R_pi = torch.einsum('sa,sa->s', policy, R)

    J = v.unsqueeze(0) @ torch.linalg.solve(torch.eye(num_states, device=device) - gamma * P_pi, R_pi.unsqueeze(1))

    return J

def tabular_feature_map(total_states, total_actions, regularizer, policy, initial_states, current_states, current_actions, next_states, rewards, gamma):
    sample_size = len(current_states)
    latent_dim = total_states * total_actions
    initial_state_sample_size = len(initial_states)

    # Create X more efficiently
    X = torch.zeros(sample_size, latent_dim, device=device)
    indices = current_states * total_actions + current_actions
    X.scatter_(1, indices.unsqueeze(1), 1)

    Y = torch.zeros(sample_size, latent_dim, device=device)
    next_state_indices = next_states[:, None] * total_actions + torch.arange(total_actions, device=device)
    Y[torch.arange(sample_size, device=device)[:, None], next_state_indices] = policy[next_states]

    W = torch.zeros(latent_dim, device=device)
    initial_state_indices = torch.tensor(initial_states, device=device)[:, None] * total_actions + torch.arange(total_actions, device=device)
    W.index_add_(0, initial_state_indices.flatten(), policy[torch.tensor(initial_states, device=device)].flatten())
    W /= initial_state_sample_size

    # Compute C_lambda, D, and E in one go
    C_lambda = X.T @ X + regularizer * torch.eye(latent_dim, device=device)
    D = X.T @ Y
    E = X.T @ rewards.unsqueeze(1)

    # Solve linear systems
    A = torch.linalg.solve(C_lambda, E).T
    M = torch.linalg.solve(C_lambda, D)

    # Compute J
    J = A @ torch.linalg.solve(torch.eye(latent_dim, device=device) - gamma * M, W)

    return J

class VectorizedAccumulatedData:
    def __init__(self, max_size=int(N*10000*0.15), device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.max_size = max_size
        self.device = device
        self.transition_counts = None
        self.reward_total = None
        self.reward_count = None
        self.initial_state_count = None
        self.initial_states = deque(maxlen=max_size)
        self.states = deque(maxlen=max_size)
        self.actions = deque(maxlen=max_size)
        self.rewards = deque(maxlen=max_size)
        self.next_states = deque(maxlen=max_size)
        self.total_steps = 0

    def update(self, transition_counts, reward_total, reward_count, initial_state_count, initial_states, trajectories, steps):
        # Update counts and totals
        print(f"Accumulated Data Size: {self.total_steps}")
        if self.transition_counts is None:
            self.transition_counts = transition_counts.to(self.device)
            self.reward_total = reward_total.to(self.device)
            self.reward_count = reward_count.to(self.device)
            self.initial_state_count = initial_state_count.to(self.device)
        else:
            self.transition_counts += transition_counts.to(self.device)
            self.reward_total += reward_total.to(self.device)
            self.reward_count += reward_count.to(self.device)
            self.initial_state_count += initial_state_count.to(self.device)

        # Update initial states
        self.initial_states.extend(initial_states)

        # Vectorized update of trajectory data
        states, actions, rewards, next_states = zip(*[step for traj in trajectories for step in traj])
        self.states.extend(states)
        self.actions.extend(actions)
        self.rewards.extend(rewards)
        self.next_states.extend(next_states)

        self.total_steps += steps

        # Trim data if necessary
        if self.total_steps > self.max_size:
            excess = self.total_steps - self.max_size
            for _ in range(excess):
                self.states.popleft()
                self.actions.popleft()
                self.rewards.popleft()
                self.next_states.popleft()
            self.total_steps = self.max_size

    def get_data(self):
        return (
            self.transition_counts,
            self.reward_total,
            self.reward_count,
            self.initial_state_count,
            list(self.initial_states),
            torch.tensor(list(self.states), device=self.device),
            torch.tensor(list(self.actions), device=self.device),
            torch.tensor(list(self.rewards), device=self.device),
            torch.tensor(list(self.next_states), device=self.device)
        )

    def process_trajectories(self):
        return (
            torch.tensor(list(self.states), device=self.device),
            torch.tensor(list(self.actions), device=self.device),
            torch.tensor(list(self.rewards), device=self.device),
            torch.tensor(list(self.next_states), device=self.device)
        )

class CustomAlgorithm:
    def __init__(self, env, method='tabular', batch_size=5037, epochs_per_batch=79, lr=0.001, max_accumulated_steps=10000, eval_episodes=10):
        self.env = env
        self.method = method
        self.batch_size = batch_size
        self.epochs_per_batch = epochs_per_batch
        self.lr = lr
        self.max_accumulated_steps = max_accumulated_steps
        self.eval_episodes = eval_episodes

        self.total_states = get_num_states(env)
        self.total_actions = env.action_space.n
        self.gamma = 0.99
        self.regularizer = 0.01

        self.theta = torch.nn.Parameter(torch.ones(self.total_states, self.total_actions, device=device) / self.total_actions)
        self.optimizer = optim.Adam([self.theta], lr=self.lr)

        self.accumulated_data = VectorizedAccumulatedData(max_size=self.max_accumulated_steps, device=device)
        self.performance_history = []

    def evaluate_policy(self):
        total_rewards = []
        for _ in range(self.eval_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            while not done:
                action, _ = self.predict(state, deterministic=True)
                next_state, reward, done, _, _ = self.env.step(action)
                episode_reward += reward
                state = next_state
            total_rewards.append(episode_reward)
        return np.mean(total_rewards)

    def learn(self, total_timesteps):
        steps_taken = 0
        while steps_taken < total_timesteps:
            # Data collection phase
            with torch.no_grad():
                policy = torch.nn.functional.softmax(self.theta, dim=1)
                new_data = sample_steps(self.env, policy, self.batch_size, max_steps_per_episode=200)
            self.accumulated_data.update(*new_data)
            steps_taken += new_data[-1]

            # Get accumulated data
            transition_counts, reward_total, reward_count, initial_state_count, initial_states, states, actions, rewards_sample, next_states = self.accumulated_data.get_data()

            v = normalise_initial(initial_state_count.float())
            R = torch.div(reward_total, reward_count.where(reward_count != 0, torch.tensor(1.0, device=device)))
            P = normalize_slices(transition_counts.float())

            # Policy optimization phase
            for _ in range(self.epochs_per_batch):
                self.optimizer.zero_grad()
                policy = torch.nn.functional.softmax(self.theta, dim=1)

                if self.method == 'counting':
                    J = compute_J_counting(self.env, policy, v, R, P, self.gamma)
                elif self.method == 'tabular':
                    J = tabular_feature_map(self.total_states, self.total_actions, self.regularizer, policy,
                                            initial_states, states, actions, next_states, rewards_sample, self.gamma)
                else:
                    raise ValueError("method must be either 'counting' or 'tabular'")

                loss = -J
                loss.backward()
                self.optimizer.step()

            # Evaluate policy after each batch
            avg_reward = self.evaluate_policy()
            self.performance_history.append((steps_taken, avg_reward))

        return self

    def predict(self, observation, state=None, deterministic=False):
        with torch.no_grad():
            policy = torch.nn.functional.softmax(self.theta, dim=1)
            if deterministic:
                action = policy[observation].argmax().item()
            else:
                action = torch.multinomial(policy[observation], 1).item()
        return action, state

def custom_algorithm(env, method='tabular', **kwargs):
    return CustomAlgorithm(env, method=method, **kwargs)



# Run the algorithm multiple times and collect results
num_runs = 10
all_results = []
method_name = 'tabular'  # THIS IS THE ONLY PLACE YOU HAVE TO CHANGE THE METHOD

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")
    algo = custom_algorithm(env, method=method_name, eval_episodes=10)
    algo.learn(total_timesteps=200000)
    print(algo.performance_history)
    all_results.append(algo.performance_history)

# Process results
step_sizes = [result[0] for result in all_results[0]]  # Assuming all runs have the same step sizes
averaged_rewards = []

for i in range(len(step_sizes)):
    rewards_at_step = [run[i][1] for run in all_results]
    avg_reward = np.mean(rewards_at_step)
    averaged_rewards.append(avg_reward)

# Create the final list of tuples (step_size, averaged_reward)
final_results = list(zip(step_sizes, averaged_rewards))

print(final_results)