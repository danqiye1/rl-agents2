import torch
import pickle
import numpy as np
from copy import deepcopy
from tqdm import tqdm
from utils import ReplayMemory, EpsilonScheduler

from pdb import set_trace as bp

default_scheduler = EpsilonScheduler()

class DQNAgent:
    """
    Implementation of Deep Q Learning Agent.
    """
    def __init__(self, env, policy_model, target_model, epsilon_scheduler=default_scheduler, buffer_size=1000000, lr=0.00025):
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

        # Replay buffer
        # Initialize replay buffer
        self.buffer = ReplayMemory(buffer_size)

        # Cummulative number of samples for each episode
        self.num_samples = []

        # Rewards for each episode
        self.rewards = []

        # Keep track of the number of frames and episodes
        self.episodes = 0
        self.num_frames = 0

        # Initialize a epsilon scheduler
        self.epsilon_scheduler = epsilon_scheduler

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

    def select_action(self, obs, epsilon):
        """ Function to select action using epsilon greedy policy """
        obs_tensor = self.numpy_to_tensor(obs, has_batch_size=True).to(self.device)
        sample = np.random.rand()

        if sample > epsilon:
            with torch.no_grad():
                q_value = self.policy_model(obs_tensor)
            return torch.argmax(q_value).item()
        else:
            return self.env.action_space.sample()

    def train_model(self, batch_size, gamma=0.9):
        """ Train the model one step """

        if len(self.buffer) < batch_size:
            # Not enough data yet
            return

        # Sample from ReplayBuffer
        states, actions, next_states, rewards = self.buffer.sample(batch_size)
        states, actions, next_states, rewards = states.to(self.device), actions.to(self.device), next_states.to(self.device), rewards.to(self.device)

        # Calculate estimated Q(s,a) based on policy network
        estimated_q = self.policy_model(states).gather(0, actions)
        
        # Calculate expected V(s_prime) based on old target network, the more stationary network.
        expected_v = torch.max(self.target_model(next_states), dim=1)[0].detach()
        target = expected_v * gamma + rewards

        loss = self.criterion(estimated_q, target.unsqueeze(1))

        # Optimize the model with gradient clipping
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

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
            # if len(obs.shape) == 2:
            #     # Grayscale image lacking channel dimension needs to be expanded
            #     obs = np.expand_dims(obs, 2)

            done = False
            iter = 0
            cum_reward = 0.0
            cum_samples = 0

            while not done:

                # Sample an action and take one step
                action = self.select_action(obs, epsilon=self.epsilon_scheduler(self.num_frames))
                action = np.asarray([action], dtype=np.int64)
                obs_prime, reward, done, info = self.env.step(action)

                # if len(obs_prime.shape) == 2:
                #     # Grayscale image lacking channel dimension needs to be expanded
                #     obs_prime = np.expand_dims(obs_prime, 2)

                # Decide if render is required for debugging and visualisation
                if render:
                    self.env.render()

                self.buffer.push(obs, action, obs_prime, reward)

                # Train the model with batch size
                # Higher batch size may cause cuda OOM
                self.train_model(batch_size)
                cum_samples += batch_size

                obs = obs_prime
                cum_reward += reward
                iter += 1
                self.num_frames += 1

                # Update target model according to update freq
                if self.num_frames % update_freq == 0:
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                    self.save(f"checkpoints/step-{self.num_frames}")
                    tqdm.write("Model saved!")
            
            self.rewards.append(cum_reward)
            self.num_samples.append(cum_samples)
            self.episodes += 1

            # Output results
            tqdm.write(f"Reward: {sum(self.rewards)/len(self.rewards)}, Steps: {self.num_frames}, Epsilon: {self.epsilon_scheduler(self.num_frames)}")

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
            pickle.dump(self.num_samples, fp)
            pickle.dump(self.num_frames, fp)

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
            self.num_samples = pickle.load(fp)
            self.num_frames = pickle.load(fp)

            # Check for errors in length
            assert len(self.rewards) == len(self.num_samples), \
                f"Mismatched length self.rewards is {len(self.rewards)} \
                but self.num_samples is {len(self.num_samples)}!"

            self.episodes = len(self.rewards)
