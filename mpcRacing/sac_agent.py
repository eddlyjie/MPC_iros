import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import os
import config

CHECK_POINT = config.CKPT

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        
        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]
        
        return states, actions, rewards, states_, dones

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name='critic', chkpt_dir=CHECK_POINT):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac.pth') # Removed: filename is now dynamic

        self.fc1 = nn.Linear(self.input_dims + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q1 = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        q1 = self.q1(x)
        return q1

    def save_checkpoint(self, suffix):
        """Saves the checkpoint with a specific suffix."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        filename = f"{self.model_name}_sac_{suffix}.pth"
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.state_dict(), save_path)

    def load_checkpoint(self, suffix):
        """Loads the checkpoint with a specific suffix."""
        filename = f"{self.model_name}_sac_{suffix}.pth"
        load_path = os.path.join(self.checkpoint_dir, filename)
        self.load_state_dict(torch.load(load_path, map_location=self.device))

class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, n_actions, action_scale, action_bias, 
                 fc1_dims=256, fc2_dims=256, name='actor', chkpt_dir=CHECK_POINT):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac.pth') # Removed: filename is now dynamic
        
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.action_scale = torch.tensor(action_scale, dtype=torch.float32).to(self.device)
        self.action_bias = torch.tensor(action_bias, dtype=torch.float32).to(self.device)
        
        self.to(self.device)

    def forward(self, state):
        # change to leaky relu
        prob = F.leaky_relu(self.fc1(state))
        prob = F.leaky_relu(self.fc2(prob))

        mu = self.mu(prob)
        log_sigma = self.sigma(prob)
        log_sigma = torch.clamp(log_sigma, min=-20, max=2) # Clamp for numerical stability
        
        return mu, log_sigma

    def sample_normal(self, state, reparameterize=True):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
        
        probabilities = Normal(mu, sigma)
        
        if reparameterize:
            actions = probabilities.rsample()  # Use rsample for backpropagation
        else:
            actions = probabilities.sample()
            
        action = torch.tanh(actions) # Squash action to [-1, 1]
        
        scaled_action = action * self.action_scale + self.action_bias
        
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(self.action_scale * (1 - action.pow(2)) + self.reparam_noise)
        log_probs = log_probs.sum(1, keepdim=True)
        
        return scaled_action, log_probs
    
    def sample_evaluate(self, state, reparameterize=True):
        mu, log_sigma = self.forward(state)
        sigma = log_sigma.exp()
            
        action = torch.tanh(mu) # Squash action to [-1, 1]
        
        scaled_action = action * self.action_scale + self.action_bias
        
        return scaled_action, None

    def save_checkpoint(self, suffix):
        """Saves the checkpoint with a specific suffix."""
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        
        filename = f"{self.model_name}_sac_{suffix}.pth"
        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.state_dict(), save_path)

    def load_checkpoint(self, suffix):
        """Loads the checkpoint with a specific suffix."""
        filename = f"{self.model_name}_sac_{suffix}.pth"
        load_path = os.path.join(self.checkpoint_dir, filename)
        self.load_state_dict(torch.load(load_path, map_location=self.device))


class SACAgent:
    def __init__(self, env, alpha=0.0003, beta=0.0003,
                 tau=0.005, reward_scale=2, gamma=0.99,
                 max_size=1000000, fc1_dims=256, fc2_dims=256,
                 batch_size=256):
        
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_scale = reward_scale
        self.save_counter = 0 # Added counter for unique save files

        self.input_dims = env.observation_space.shape
        self.n_actions = env.action_space.shape[0]
        # Handle 1D action space scales/biases correctly
        if np.isscalar(env.action_space.high):
             self.action_scale = (env.action_space.high - env.action_space.low)/2
             self.action_bias = (env.action_space.low + env.action_space.high) /2
        else:
            self.action_scale = (env.action_space.high - env.action_space.low)/2.0
            self.action_bias = (env.action_space.low + env.action_space.high) / 2.0

        self.memory = ReplayBuffer(max_size, self.input_dims, self.n_actions)
        
        # Ensure input_dims[0] is used, as input_dims is a shape tuple
        actor_input_dims = self.input_dims[0]
        critic_input_dims = self.input_dims[0]

        self.actor = ActorNetwork(alpha, actor_input_dims, self.n_actions,
                                  action_scale=self.action_scale, action_bias=self.action_bias,
                                  fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='actor')
        
        self.critic_1 = CriticNetwork(beta, critic_input_dims, self.n_actions,
                                      fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='critic_1')
        self.critic_2 = CriticNetwork(beta, critic_input_dims, self.n_actions,
                                      fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='critic_2')
        
        self.target_critic_1 = CriticNetwork(beta, critic_input_dims, self.n_actions,
                                             fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='target_critic_1')
        self.target_critic_2 = CriticNetwork(beta, critic_input_dims, self.n_actions,
                                             fc1_dims=fc1_dims, fc2_dims=fc2_dims, name='target_critic_2')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation, evaluate=False):
        state = torch.Tensor([observation]).to(self.actor.device)
        if evaluate:
            actions, _ = self.actor.sample_evaluate(state, reparameterize=False)
        else:
            actions, _ = self.actor.sample_normal(state, reparameterize=False)
        return actions.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        # Update target critic 1
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        # Update target critic 2
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        state, action, reward, new_state, done = self.memory.sample_buffer(self.batch_size)
        
        device = self.actor.device
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).to(device)
        new_state = torch.tensor(new_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.bool).to(device)

        # --- Critic Loss ---
        with torch.no_grad():
            new_actions, new_log_probs = self.actor.sample_normal(new_state, reparameterize=True)
            q1_target = self.target_critic_1.forward(new_state, new_actions)
            q2_target = self.target_critic_2.forward(new_state, new_actions)
            q_target = torch.min(q1_target, q2_target) - self.reward_scale * new_log_probs # Using scaled entropy
            q_target[done] = 0.0
            y = reward + self.gamma * q_target.view(-1)

        q1 = self.critic_1.forward(state, action).view(-1)
        q2 = self.critic_2.forward(state, action).view(-1)
        
        critic_1_loss = F.mse_loss(q1, y)
        critic_2_loss = F.mse_loss(q2, y)
        critic_loss = critic_1_loss + critic_2_loss
        
        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        # --- Actor Loss ---
        # Freeze critic networks for actor update
        for p in self.critic_1.parameters(): p.requires_grad = False
        for p in self.critic_2.parameters(): p.requires_grad = False

        actions, log_probs = self.actor.sample_normal(state, reparameterize=True)
        q1_pi = self.critic_1.forward(state, actions)
        q2_pi = self.critic_2.forward(state, actions)
        q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = (self.reward_scale * log_probs.view(-1) - q_pi.view(-1)).mean()
        
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        # Unfreeze critic networks
        for p in self.critic_1.parameters(): p.requires_grad = True
        for p in self.critic_2.parameters(): p.requires_grad = True

        # --- Update Target Networks ---
        self.update_network_parameters()

    def save_models(self):
        """Saves all models with a unique versioned suffix."""
        print(f'... saving models (save #{self.save_counter}) ...')
        suffix = f"save{self.save_counter}"
        self.actor.save_checkpoint(suffix)
        self.critic_1.save_checkpoint(suffix)
        self.critic_2.save_checkpoint(suffix)
        self.target_critic_1.save_checkpoint(suffix)
        self.target_critic_2.save_checkpoint(suffix)
        self.save_counter += 1 # Increment for the next save

    def load_models(self, suffix):
        """Loads all models from a specific versioned suffix."""
        print(f'... loading models (save {suffix}) ...')
        self.actor.load_checkpoint(suffix)
        self.critic_1.load_checkpoint(suffix)
        self.critic_2.load_checkpoint(suffix)
        self.target_critic_1.load_checkpoint(suffix)
        self.target_critic_2.load_checkpoint(suffix)
