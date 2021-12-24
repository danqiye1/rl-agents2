import gym
import torch
import numpy as np
from tqdm import tqdm
from networks import models
from ReplayBuffer.buffer import ReplayMemory

from pdb import set_trace as bp

# Some global parameters
env = gym.make("Breakout-v4")
observation = env.reset()
done = False
update_freq = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Get observation dimensions and size of action space
height, width, nch = observation.shape
n_actions = env.action_space.n

# Initialize 2 identical network
policy_model = models.QNetwork(height, width, nch, n_actions).to(device)
target_model = models.QNetwork(height, width, nch, n_actions).to(device)
target_model.load_state_dict(policy_model.state_dict())
target_model.eval() # Set's tartget model in evaluation mode since we are not training this

# Initialize Optimizer and loss criterion (Huber Loss)
criterion = torch.nn.SmoothL1Loss()
optimizer = torch.optim.RMSprop(policy_model.parameters())

# Initialize replay buffer
buffer = ReplayMemory(10000)

def numpy_to_tensor(obs, has_batch_size=False):
    """ Helper function to convert numpy to torch tensor """
    if has_batch_size:
        obs_tensor = (torch.from_numpy(obs)
            .permute(0,3,1,2)
            .type(torch.FloatTensor))
    else:
        obs_tensor = (torch.from_numpy(obs)
            .permute(2,0,1)
            .type(torch.FloatTensor)
            .unsqueeze(0))

    return obs_tensor

def select_action(obs, epsilon):
    """ Function to select action using epsilon greedy policy """
    obs_tensor = numpy_to_tensor(obs).to(device)
    sample = np.random.rand()

    if sample > epsilon:
        with torch.no_grad():
            q_value = policy_model(obs_tensor)
        return torch.argmax(q_value).item()
    else:
        return env.action_space.sample()

def train_model(batch_size, gamma=0.9):
    """ Train the model one step """

    if len(buffer) < batch_size:
        # Not enough data yet
        return

    states, actions, next_states, rewards = buffer.sample(batch_size)
    states, actions, next_states, rewards = states.to(device), actions.to(device), next_states.to(device), rewards.to(device)

    # Calculate estimated Q(s,a) based on policy network
    estimated_q = policy_model(states).gather(1, actions.unsqueeze(-1))
    
    # Calculate expected V(s_prime) based on old target network, the more stationary network.
    expected_v = torch.max(target_model(next_states), dim=1)[0].detach()
    target = expected_v * gamma + rewards

    loss = criterion(estimated_q, target.unsqueeze(1))

    # Optimize the model with gradient clipping
    optimizer.zero_grad()
    loss.backward()
    for param in policy_model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

# To Do: Write the training loop and figure out the sample efficiencies
num_episodes = 50
for episode in tqdm(range(num_episodes)):
    obs = env.reset()

    while not done:
        # Sample an action and take one step
        action = select_action(obs, epsilon=0.3)
        obs_prime, reward, done, info = env.step(action)

        buffer.push(obs, action, obs_prime, reward)

        # Train the model
        train_model(32)

        obs = obs_prime

    # Done with 1 episode.
    # Transfer states of trained policy model to target model ocassionally
    # for expected Q values
    if episode % update_freq == 0:
        target_model.load_state_dict(policy_model.state_dict())

    # Reset done
    done = False