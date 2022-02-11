""" Let's make DQN Agent learn pong """
import gym
import argparse
from agents import DQNAgent
from network import MinhDQN

# Command line parser
parser = argparse.ArgumentParser(description="Atari Pong parameters")
parser.add_argument("--train", dest="is_train_mode", action="store_true")
parser.add_argument("--eval", dest="is_train_mode", action="store_false")
parser.add_argument("--load-model", '-l', dest="model_path", default=None, type=str)
args = parser.parse_args()

# Instantiate environment
env = gym.make("PongDeterministic-v4")
observation = env.reset()

# Get observation dimensions and size of action space
_, height, width = observation.shape
n_actions = env.action_space.n

# Initialize critic model and agent
policy_model = MinhDQN(4, n_actions)
target_model = MinhDQN(4, n_actions)
agent = DQNAgent(env, policy_model, target_model)

# Load pretrained model if path is given
if args.model_path:
    agent.load(args.model_path)

# Either train or evaluate the model
if args.is_train_mode:
    agent.train(render=True)
else:
    agent.eval_model()