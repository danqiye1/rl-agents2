""" Let's make DQN Agent learn pong """

from stable_baselines3.common.cmd_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

from agents import DQNAgent
from network import MinhDQN

# Instantiate environment
env = make_atari_env("PongNoFrameskip-v4")
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

observation = env.reset()

# Get observation dimensions and size of action space
_, height, width, nch = observation.shape
n_actions = env.action_space.n

# Initialize critic model and agent
model = MinhDQN(nch, n_actions)
agent = DQNAgent(env, model)

# Training loop for 100 episodes
agent.train(render=True)