""" Let's make DQN Agent learn pong """

import gym

from agents import DQNAgent
from network import MinhDQN

from pdb import set_trace as bp

# Some global parameters
env = gym.make("PongNoFrameskip-v0")
env = gym.wrappers.AtariPreprocessing(env, grayscale_newaxis=True)

observation = env.reset()

# Get observation dimensions and size of action space
height, width, nch = observation.shape
n_actions = env.action_space.n

# Initialize critic model and agent
model = MinhDQN(nch, n_actions)
agent = DQNAgent(env, model)

# Training loop for 100 episodes
agent.load("checkpoints/episode90")
agent.train(render=True)