import random
import torch
import numpy as np
from collections import namedtuple, deque

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, state, action, next_state, reward):
        """Save a transition"""
        self.memory.append(Transition(state, action, next_state, reward))

    def sample(self, batch_size):
        """ Sample a batch of transitions and convert to numpy arrays for training """
        transitions = random.sample(self.memory, batch_size)
        states, actions, next_states, rewards = [], [], [], []

        # Unpack the transition
        for state, action, next_state, reward in transitions:
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)

        return (
            torch.from_numpy(np.asarray(states)).type(torch.FloatTensor).permute(0,3,1,2), 
            torch.LongTensor(actions), 
            torch.from_numpy(np.asarray(next_states)).type(torch.FloatTensor).permute(0,3,1,2), 
            torch.Tensor(rewards)
        )

    def __len__(self):
        return len(self.memory)