import random
import torch
import numpy as np
from collections import namedtuple, deque

from pdb import set_trace as bp

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward, done):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward, done))

    def sample(self, batch_size):
        """ Sample a batch of transitions and convert to numpy arrays for training """
        transitions = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards, termination = [], [], [], [], []

        # Unpack the transition
        for state, action, next_state, reward, done in transitions:
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            termination.append(done)

        return (
            torch.Tensor(np.array(states)), 
            torch.LongTensor(np.array(actions)), 
            torch.Tensor(np.array(next_states)), 
            torch.Tensor(np.array(rewards)),
            torch.LongTensor(np.array(termination))
        )

    def __len__(self):
        return len(self.memory)