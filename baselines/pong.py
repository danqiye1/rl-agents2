import argparse

from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
from stable_baselines3 import DQN

# Instantiate environment
env = make_atari_env("PongNoFrameskip-v4")
# Frame-stacking with 4 frames
env = VecFrameStack(env, n_stack=4)

# Command line parser
parser = argparse.ArgumentParser(description="Atari Pong parameters")
parser.add_argument("--train", dest="is_train_mode", action="store_true")
parser.add_argument("--eval", dest="is_train_mode", action="store_false")
parser.add_argument("--load-model", '-l', dest="model_path", default=None, type=str)
args = parser.parse_args()



# Load pretrained model if path is given
if args.model_path:
    model = DQN.load(args.model_path)
else:
    model = DQN("CnnPolicy", env, verbose=1)

if args.is_train_mode:
    model.learn(total_timesteps=1000000, log_interval=10)
    model.save("baselines/pong-dqn")
else:
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        if done:
            obs = env.reset()