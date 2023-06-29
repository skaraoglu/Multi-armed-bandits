import sys
import matplotlib.pyplot as plt
import numpy as np

# ...

# Read the command-line arguments for number of arms and number of episodes
if len(sys.argv) < 3:
    print("Usage: python bandits.py num_arms num_episodes")
    sys.exit(1)

num_arms = int(sys.argv[1])
num_episodes = int(sys.argv[2])

import matplotlib.pyplot as plt
import numpy as np

# Create a Bandit class representing the multi-armed bandit
class Bandit:
    def __init__(self, num_arms):
        # Initialize the Bandit object with the number of arms
        self.num_arms = num_arms
        # Generate true rewards for each arm from a standard normal distribution
        self.true_rewards = np.random.normal(0, 1, num_arms)
        
    def pull_arm(self, arm_index):
        # Simulate pulling an arm of the bandit and returning the reward
        # The reward is sampled from a normal distribution with mean equal to the true reward of the arm
        # and standard deviation of 1
        reward = np.random.normal(self.true_rewards[arm_index], 1)
        reward = 1 / (1 + np.exp(-reward))  # Apply sigmoid transformation
        return reward

def epsilon_greedy(q_values, n_pulls, c, prior_wins, prior_losses, epsilon=0.1):
    num_arms = len(q_values)
    if np.random.random() < epsilon:
        # Explore: choose a random arm
        action = np.random.randint(num_arms)
    else:
        # Exploit: choose the arm with the highest q-value
        action = np.argmax(q_values)
    return action, None

def ucb(q_values, n_pulls, prior_wins, prior_losses, c=1.0):
    num_arms = len(q_values)
    exploration_term = np.zeros(num_arms)
    # Calculate the exploration term
    non_zero_n_pulls = n_pulls[n_pulls > 0]
    exploration_term[n_pulls > 0] = c * np.sqrt(np.log(sum(n_pulls[n_pulls > 0]) + 1e-6) / non_zero_n_pulls + 1e-6)
    exploration_term[n_pulls == 0] = float('inf')
    ucb_values = q_values + exploration_term
    # Select the arm with the highest UCB value
    action = np.argmax(ucb_values)
    return action, None

def thompson_sampling(q_values, n_pulls, c, prior_wins, prior_losses):
    # Set a small epsilon value to avoid division by zero
    epsilon = 1e-6
    # Calculate the alpha and beta parameters for the Beta distribution
    alpha = np.maximum(prior_wins, epsilon)
    beta = np.maximum(prior_losses, epsilon)
    # Sample success rates from the Beta distribution
    success_rates = np.random.beta(alpha, beta)  # Remove the +1 term
    # Select the arm with the highest success rate
    action = np.argmax(success_rates)
    return action, None

def bayesian_greedy(q_values, n_pulls, c, prior_wins=1, prior_losses=1):
    num_arms = len(q_values)
    posterior_wins = prior_wins + np.sum(n_pulls, axis=0)
    posterior_losses = prior_losses + np.sum(1 - n_pulls, axis=0)
    # Calculate the mean win rate for each arm
    win_rates = posterior_wins / (posterior_wins + posterior_losses)
    # Select the arm with the highest mean win rate
    action = np.argmax(win_rates)
    
    return action, None

def ucb_horizon_aware(q_values, n_pulls, prior_wins, prior_losses, horizon=1000, c=1.0):
    num_arms = len(q_values)
    exploration_term = np.zeros(num_arms)
    # Calculate the exploration term for arms with non-zero number of pulls
    exploration_term[n_pulls > 0] = c * np.sqrt(2 * np.log(horizon) / n_pulls[n_pulls > 0])
    exploration_term[n_pulls == 0] = float('inf')
    ucb_values = q_values + exploration_term
    # Select the arm with the highest UCB value
    action = np.argmax(ucb_values)
    return action, None

# Set up the multi-armed bandit experiment variables
true_means = np.random.normal(0, 1, num_arms)
epsilon = 0.1
c = 1.0

# Initialize the bandit and q-values
bandit = Bandit(num_arms)
q_values = np.zeros(num_arms)
n_pulls = np.zeros(num_arms)

# Define the list of algorithms to test
algorithms = [
    (epsilon_greedy, 'Epsilon-Greedy'),
    (ucb, 'UCB'),
    (thompson_sampling, 'Thompson Sampling'),
    (bayesian_greedy, 'Bayesian Greedy'),
    (ucb_horizon_aware, 'HA-UCB')
]

# Initialize a dictionary to store the cumulative rewards for each algorithm
cumulative_rewards = {algorithm_name: [] for _, algorithm_name in algorithms}
average_rewards = {algorithm_name: [] for _, algorithm_name in algorithms}
# Run the multi-armed bandit experiment for each algorithm
for algorithm_func, algorithm_name in algorithms:
    rewards = []
    n_pulls = np.zeros(num_arms)
    q_values = np.zeros(num_arms)
    prior_wins = 1
    prior_losses = 1
    for episode in range(num_episodes):
        action, _ = algorithm_func(q_values, n_pulls, c, prior_wins, prior_losses)
        reward = bandit.pull_arm(action)
        rewards.append(reward)
        n_pulls[action] += 1
        q_values[action] += (reward - q_values[action]) / n_pulls[action]
        average_reward = np.mean(rewards)
        average_rewards[algorithm_name].append(average_reward)
    cumulative_rewards[algorithm_name] = np.cumsum(rewards)

# Plot the results of the multi-armed bandit experiment for different algorithms
plt.figure(figsize=(12,6))
for algorithm_name, cumulative_reward in cumulative_rewards.items():
    plt.plot(cumulative_reward, label=algorithm_name)

plt.xlabel('Episode')
plt.ylabel('Cumulative Reward')
plt.title('Multi-Armed Bandit Experiment')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
for algorithm_name, avg_rewards in average_rewards.items():
    plt.plot(avg_rewards, label=algorithm_name)

plt.xlabel('Episode')
plt.ylabel('Average Reward')
plt.title('Average Reward over Episodes')
plt.legend()
plt.show()
