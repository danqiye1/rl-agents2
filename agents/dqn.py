import cv2
import torch
import pickle
import numpy as np
from tqdm import tqdm
from collections import deque
from utils import ReplayMemory, EpsilonScheduler
from torch.utils.tensorboard import SummaryWriter

from pdb import set_trace as bp

default_scheduler = EpsilonScheduler()

class DQNAgent:
    """
    Implementation of Deep Q Learning Agent.
    """
    def __init__(self, env, policy_model, target_model, epsilon_scheduler=default_scheduler, buffer_size=100000, lr=0.00025):
        """
        :param env: An env interface following OpenAI gym specifications.
        :param policy_model: A model to estimate q from input observations.
        :param target_model: A model to estimate target q from input observataions.
        :type model: torch.nn.Module

        :param epsilon_scheduler: An object to schedule the epsilon decay rate.
        :param buffer_size: Size of replay buffer.
        :param lr: Learning rate for RMSProp.
        """
        # Internal device mapper
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Internal model representations
        assert policy_model is not target_model, "Target model and policy model should not reference the same object!"
        self.policy_model = policy_model.to(self.device)
        self.target_model = target_model.to(self.device)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        # Internal optimizers
        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.RMSprop(self.policy_model.parameters(), lr=lr)

        # Interface with environment
        self.env = env

        # State size for breakout env. SS images (210, 160, 3). Used as input size in network
        self.state_size_h = env.observation_space.shape[0]
        self.state_size_w = env.observation_space.shape[1]
        self.state_size_c = env.observation_space.shape[2]

        # Activation size for breakout env. Used as output size in network
        self.action_size = env.action_space.n

        # Image pre process params
        self.target_h = 84  # Height after process
        self.target_w = 84  # Widht after process

        self.crop_dim = [20, self.state_size_h, 0, self.state_size_w]  # Cut 20 px from top to get rid of the score table
        

        # Replay buffer
        # Initialize replay buffer
        self.buffer = ReplayMemory(buffer_size)

        # Rewards for each episode
        self.rewards = deque(maxlen=100)

        # Track the cummulative max Q estimate of training:
        self.max_q = deque(maxlen=100)

        # Keep track of the number of frames and episodes
        self.episodes = 0
        self.num_frames = 0

        # Initialize a epsilon scheduler
        self.epsilon_scheduler = epsilon_scheduler

        # Initialize a tensorboard writer
        self.writer = SummaryWriter()

        self.epsilon = 1

    def numpy_to_tensor(self, obs, has_batch_size=False):
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

    def preProcess(self, image):
        """
        Process image crop resize, grayscale and normalize the images
        """
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # To grayscale
        frame = frame[self.crop_dim[0]:self.crop_dim[1], self.crop_dim[2]:self.crop_dim[3]]  # Cut 20 px from top
        frame = cv2.resize(frame, (self.target_w, self.target_h))  # Resize
        frame = frame.reshape(self.target_w, self.target_h) / 255  # Normalize

        return frame

    def select_action(self, obs, epsilon):
        """ Function to select action using epsilon greedy policy """
        # obs_tensor = self.numpy_to_tensor(obs, has_batch_size=False).to(self.device)
        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(self.device)
        sample = np.random.rand()

        if sample > epsilon:
            with torch.no_grad():
                q_value = self.policy_model(obs_tensor)
            return torch.argmax(q_value).item()
        else:
            return self.env.action_space.sample()

    def train_model(self, batch_size, gamma=0.9):
        """ 
        Train the model one step 
        
        :return: max Q value of this batch.
        """

        if len(self.buffer) < batch_size:
            # Not enough data yet
            return 0

        # Sample from ReplayBuffer
        states, actions, next_states, rewards, done = self.buffer.sample(batch_size)
        states, actions = states.to(self.device), actions.to(self.device)
        next_states, rewards, done = next_states.to(self.device), rewards.to(self.device), done.to(self.device)

        # Calculate estimated Q(s,a) based on policy network
        policy_q = self.policy_model(states)
        estimated_q = policy_q.gather(1, actions)

        # Calculate expected V(s_prime) based on old target network, the more stationary network.
        expected_v = torch.max(self.target_model(next_states), dim=1)[0].detach()
        target = rewards + gamma * expected_v * (1 - done)

        # loss = self.criterion(estimated_q, target.unsqueeze(1))
        loss = (estimated_q.squeeze() - target.detach()).pow(2).mean()

        # Optimize the model with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return torch.max(policy_q).item()

    def train(self, num_episodes=1000, batch_size=32, update_freq=10000, render=True):
        """ Main training function for Deep Q Learning 
        
        :param num_episodes: Number of episodes to train on.
        :param max_iter: Maximum number of iterations per episode.
        :param batch_size: Batch size of 1 backpropagation.
        :param update_freq: Frequency of updates for target network.
        :param epsilon: Exploration probability.
        """
        for episode in tqdm(range(num_episodes)):

            obs = self.env.reset()
            obs = self.preProcess(obs)

            obs = np.stack((obs, obs, obs, obs))

            done = False
            iter = 0
            cum_reward = 0.0

            while not done:

                # Sample an action and take one step
                # action = self.select_action(obs, epsilon=self.epsilon_scheduler(self.num_frames))
                action = self.select_action(obs, epsilon = self.epsilon)
                action = np.asarray([action], dtype=np.int64)
                obs_prime, reward, done, info = self.env.step(action)

                obs_prime = self.preProcess(obs_prime)
                obs_prime = np.stack((obs_prime, obs[0], obs[1], obs[2]))

                # Decide if render is required for debugging and visualisation
                if render:
                    self.env.render()

                self.buffer.push(obs, action, obs_prime, reward, done)

                # Train the model with batch size
                # Higher batch size may cause cuda OOM
                max_q = self.train_model(batch_size)
                self.max_q.append(max_q)
                # cum_samples += batch_size

                obs = obs_prime
                cum_reward += reward
                iter += 1
                self.num_frames += 1
                
                if self.num_frames % 1000 == 0:
                    if self.epsilon > 0.05:
                        self.epsilon *= 0.99

                # Update target model according to update freq
                if self.num_frames % update_freq == 0:
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                    self.save(f"checkpoints/step-{self.num_frames}")
                    tqdm.write("Model saved!")
            
            self.rewards.append(cum_reward)
            self.episodes += 1

            # Calculate statistics for tqdm and tensorboard
            # For tracking experiment performance
            avg_reward = sum(self.rewards) / len(self.rewards)
            avg_q = sum(self.max_q) / len(self.max_q)

            # Output results
            tqdm.write(
                f"Avg Reward: {avg_reward}, \
                Steps: {self.num_frames}, \
                Avg Q: {avg_q: .3f}, \
                Epsilon: {self.epsilon: .3f}")

            self.writer.add_scalar('Avg/Reward', avg_reward, episode)
            self.writer.add_scalar('Avg/Max Q', avg_q, episode)
            self.writer.add_scalar('Epsilon', self.epsilon, episode)

            # Reset done
            done = False

    def eval_model(self, render=True):
        """ Evaluate the target model """

        print("\nEvaluating model...")
        eval_done = False
        obs = self.env.reset()
        cum_reward = 0.0
        step = 0

        while not eval_done:
            # We don't use select model because we want to follow policy greedily with target_model
            with torch.no_grad():
                obs_tensor = self.numpy_to_tensor(obs, has_batch_size=True)
                q_val = self.target_model(obs_tensor.to(self.device))
                action = torch.argmax(q_val).item()
            
            action = np.asanyarray([action], dtype=np.int64)
            obs, reward, eval_done, _ = self.env.step(action)
            if render:
                self.env.render()
                
            step += 1

            cum_reward += reward

        print("Evaluation done!")

        return cum_reward

    def save(self, filename):
        """ Save the model during training 
        
        :param filename: Prefix of filename for saving the model and training meta-data.
        """
        # Save the current target model
        torch.save(self.target_model.state_dict(), f"{filename}-model.pt")

        # Save the performance of current target model
        with open(f"{filename}-meta.pkl", "wb") as fp:
            pickle.dump(self.rewards, fp)
            pickle.dump(self.num_frames, fp)
            pickle.dump(self.max_q, fp)

    def load(self, filename):
        """ Load the agent 
        
        :param filename: Prefix of filename for saving the model and training meta-data.
        """
        # Load the target model
        self.target_model.load_state_dict(torch.load(f"{filename}-model.pt"))
        self.policy_model.load_state_dict(self.target_model.state_dict())

        # Load the training results
        with open(f"{filename}-meta.pkl", "rb") as fp:
            self.rewards = pickle.load(fp)
            self.num_frames = pickle.load(fp)

            self.episodes = len(self.rewards)
