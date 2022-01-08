""" Let's make DQN Agent learn pong """

import gym

from agents import DQNAgent
from network import SqueezeNetQValCritic

# Some global parameters
env = gym.make("Pong-v0")
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