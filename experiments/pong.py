""" Let's make DQN Agent learn pong """
import gym
import argparse
from agents import DQNAgent
from network import MinhDQN
from torch.utils.tensorboard import SummaryWriter

# Agent Hyperparameters
ENV_NAME = "PongDeterministic-v4"
FRAMESKIP = 4
UPDATE_STEPS = 10000
BUFFER_SIZE = 100000
GAMMA = 0.97

EPSILON_START = 1
EPSILON_END = 0.05
EPSILON_DECAY = 0.99
EPSILON_DECAY_FREQ = 1000

# Model Hyperparameters
LEARNING_RATE = 0.00025

# Training Hyperparameters
NUM_EPISODES = 1000
BATCH_SIZE = 32
RENDER = True
LOG_DIR = "runs/dqn-pong"

# Command line parser
parser = argparse.ArgumentParser(description="Atari Pong parameters")
parser.add_argument("--train", dest="is_train_mode", action="store_true")
parser.add_argument("--eval", dest="is_train_mode", action="store_false")
parser.add_argument("--load-model", '-l', dest="model_path", default=None, type=str)
args = parser.parse_args()

# Instantiate environment
env = gym.make(ENV_NAME)
observation = env.reset()

# Get observation dimensions and size of action space
_, height, width = observation.shape
n_actions = env.action_space.n

tensorboard_writer = SummaryWriter(LOG_DIR)

# Initialize critic model and agent
policy_model = MinhDQN(FRAMESKIP, n_actions)
target_model = MinhDQN(FRAMESKIP, n_actions)
agent = DQNAgent(
    env, policy_model, target_model, EPSILON_START, EPSILON_END, EPSILON_DECAY, EPSILON_DECAY_FREQ,
    GAMMA, BUFFER_SIZE, LEARNING_RATE, tensorboard_writer
)

# Load pretrained model if path is given
if args.model_path:
    agent.load(args.model_path)

# Either train or evaluate the model
if args.is_train_mode:
    agent.train(NUM_EPISODES, BATCH_SIZE, UPDATE_STEPS, RENDER)
else:
    agent.eval_model()