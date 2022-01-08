import gym

from agents import DQNAgent
from network import SqueezeNetQValCritic

# Some global parameters
env = gym.make("Breakout-v0")
observation = env.reset()
done = False
update_freq = 10

# Get observation dimensions and size of action space
height, width, nch = observation.shape
n_actions = env.action_space.n

# Initialize critic model and agent
model = SqueezeNetQValCritic(height, width, nch, n_actions)
agent = DQNAgent(env, model)

# Training loop for 1000 episodes
agent.train(num_episodes=1000)

# To Do:
# 0. Breakout gets stuck if the action is not fire.
# 1. Concatenate history of 4 frames
# 2. Implement max steps for evaluation (Implemented)
# 3. Implement decaying epsilon
# 4. Use Kaiming Initialization (Implemented)