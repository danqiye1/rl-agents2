""" Let's make DQN Agent learn pong """
import gym
import argparse
import wandb
from agents import DQNAgent
from network import MinhDQN

# Command line parser
parser = argparse.ArgumentParser(description="Atari Breakout parameters")
parser.add_argument("--train", dest="is_train_mode", action="store_true")
parser.add_argument("--eval", dest="is_train_mode", action="store_false")
parser.add_argument("--load-model", '-l', dest="model_path", default=None, type=str)
parser.add_argument("--render", '-r', dest="render", action="store_true")
parser.add_argument("--resume", dest="wandb_resume", action="store_true")
args = parser.parse_args()

parameters = {
    "env": "BreakoutDeterministic-v4",
    "frameskip": 4,
    "update_steps": 10000,
    "buffer_size": 100000,
    "gamma": 0.97,
    "epsilon_start": 1,
    "epsilon_end": 0.05,
    "epsilon_decay": 0.99,
    "epsilon_decay_freq": 1000,
    "learning_rate": 0.00025,
    "num_episodes": 1000,
    "batch_size": 32
}

# Instantiate environment
env = gym.make(parameters["env"])
observation = env.reset()

# Get observation dimensions and size of action space
_, height, width = observation.shape
n_actions = env.action_space.n

# Initialize critic model and agent
policy_model = MinhDQN(parameters["frameskip"], n_actions)
target_model = MinhDQN(parameters["frameskip"], n_actions)

agent = DQNAgent(
    env, policy_model, target_model, 
    parameters["epsilon_start"], 
    parameters["epsilon_end"], 
    parameters["epsilon_decay"], 
    parameters["epsilon_decay_freq"],
    parameters["gamma"],
    parameters["buffer_size"],
    parameters["learning_rate"]
)

# Load pretrained model if path is given
if args.model_path:
    agent.load(args.model_path)

# Either train or evaluate the model
if args.is_train_mode:
    # Configure Weights and Biases
    wandb.init(project="dqn-breakout", entity=args.entity, config=parameters)
    agent.train(parameters["num_episodes"], parameters["batch_size"], parameters["update_steps"], args.render)
else:
    agent.eval_model()