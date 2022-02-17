# Reinforcement Learning Agents

I hope that this repository will be my personal framework for Deep Reinforcement Learning in future. Of course this remains to be seen.

## Current WIP:
1. Deep Q Learning on Atari Breakout.
2. Deep Q Learning on Atari Pong.

## To Do:
0. Breakout gets stuck if the action is not fire.
1. Concatenate history of 4 frames (Implemented by Gym default)
2. Implement max steps for evaluation (Implemented)
3. Implement decaying epsilon (Implemented)
4. Use Kaiming Initialization (Implemented)
5. Implement checkpoints to save agents and models (Implemented)
6. Keep track of episodes (Implemented)
7. Use original CNN architecture in 2013 paper for sanity.
8. Make env terminate after one death.

## Running Experiments
Entry points to experiments are in `experiments` folder. 

To train a pong agent:
```bash
$ python experiments/pong.py --train
```

To continue training from a model:
```bash
$ python experiments/pong.py --train -l checkpoints/step-<step_num>
```

To evaluate a model:
```bash
$ python experiments/pong.py --eval -l checkpoints/step-<step_num>
```