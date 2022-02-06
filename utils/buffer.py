import random
import torch
import numpy as np
from collections import namedtuple, deque

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
            torch.from_numpy(np.concatenate(states)).type(torch.FloatTensor).permute(0,3,1,2), 
            torch.LongTensor(np.array(actions)), 
            torch.from_numpy(np.concatenate(next_states)).type(torch.FloatTensor).permute(0,3,1,2), 
            torch.Tensor(np.concatenate(rewards)),
            torch.Tensor(np.concatenate(termination))
        )

    def __len__(self):
        return len(self.memory)