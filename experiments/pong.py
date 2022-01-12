""" Let's make DQN Agent learn pong """

import gym

from agents import DQNAgent
from network import SqueezeNetQValCritic

# Some global parameters
env = gym.make("PongNoFrameskip-v0")
env = gym.wrappers.AtariPreprocessing(env)

observation = env.reset()
done = False
update_freq = 10

# Get observation dimensions and size of action space
height, width = observation.shape
nch = 1
n_actions = env.action_space.n

# Initialize critic model and agent
model = SqueezeNetQValCritic(height, width, nch, n_actions)
agent = DQNAgent(env, model)

# Training loop for 1000 episodes
# agent.load("checkpoints/episode30")
agent.train(num_episodes=1000, render=True)