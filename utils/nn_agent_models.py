"""
Base models taken from the PTAN (PyTorch Agent Net) library by Shmuma
https://github.com/Shmuma/ptan
and https://github.com/taldatech/pytorch-ls-ddpg
"""
# imports
import torch
import torch.nn as nn
import numpy as np
import copy
import math




def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)

class CoSine(nn.Module):

    def __init__(self):
        super(CoSine, self).__init__()
        pass
        
        
    def forward(self, x):
        
        return torch.cos(x)

class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """

    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync(self, alpha):
        """
        Blend params of target net with params from the model
        :param alpha:
        """
        assert isinstance(alpha, float)
        assert 0.0 < alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        self.target_model.load_state_dict(tgt_state)


class BaseAgent:
    """
    Abstract Agent interface
    """

    def initial_state(self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__(self, states, agent_states):
        """
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :param agent_states: list of states with the same length as observations
        :return: tuple of actions, states
        """
        assert isinstance(states, list)
        assert isinstance(agent_states, list)
        assert len(agent_states) == len(states)

        raise NotImplementedError


class Actor(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Actor, self).__init__()

        self.fc1 = nn.Sequential(nn.Linear(obs_size, 400),
                                 nn.ReLU()
                                 )
       
        self.fc2 = nn.Linear(400, 400)
        self.relu_2 = nn.ReLU()
        self.fc3 = nn.Linear(400, act_size)
        self.tanh_3 = nn.Tanh()

        stdv = 1. / math.sqrt(self.fc1[0].weight.size(1))
        self.fc1[0].weight.data.uniform_(-stdv,stdv)
        self.fc1[0].weight.requires_grad = False
        if self.fc1[0].bias is not None:
            self.fc1[0].bias.data.uniform_(-stdv, stdv)
        self.fc1[0].bias.requires_grad = False

        stdv = 1. / math.sqrt(self.fc2.weight.size(1))
        self.fc2.weight.data.uniform_(-stdv,stdv)
        self.fc2.weight.requires_grad = False
        if self.fc2.bias is not None:
            self.fc2.bias.data.uniform_(-stdv, stdv)
        self.fc2.bias.requires_grad = False
        

    def forward(self, x):
        return self.tanh_3(self.fc3(self.relu_2(self.fc2(self.fc1(x)))))

    def forward_to_last_hidden(self, x):
        return self.relu_2(self.fc2(self.fc1(x)))


class Critic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(Critic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU()
        )

        self.out_fc1 = nn.Linear(400 + act_size, 300)
        self.relu_1 = nn.ReLU()

        self.out_fc2 = nn.Linear(300, 1)
        stdv = 1. / math.sqrt(self.obs_net[0].weight.size(1))
        self.obs_net[0].weight.data.uniform_(-stdv,stdv)
        self.obs_net[0].weight.requires_grad = False
        if self.obs_net[0].bias is not None:
            self.obs_net[0].bias.data.uniform_(-stdv, stdv)
        self.obs_net[0].bias.requires_grad = False

        stdv = 1. / math.sqrt(self.out_fc1.weight.size(1))
        self.out_fc1.weight.data.uniform_(-stdv,stdv)
        self.out_fc1.weight.requires_grad = False
        if self.out_fc1.bias is not None:
            self.out_fc1.bias.data.uniform_(-stdv, stdv)
        self.out_fc1.bias.requires_grad = False
        
       

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_fc2(self.relu_1(self.out_fc1((torch.cat([obs, a], dim=1)))))

    def forward_to_last_hidden(self, x, a):
        obs = self.obs_net(x)
        return self.relu_1(self.out_fc1((torch.cat([obs, a], dim=1))))


class AgentRANDPOL(BaseAgent):
    """
    Agent implementing Orstein-Uhlenbeck exploration process.
    # Implemented noise decaying for convergence
    """

    def __init__(self, net, device="cpu", ou_enabled=True, ou_mu=0.0, ou_teta=0.15, ou_sigma=0.2, ou_epsilon=1.0,
                 ou_decay_steps=500000, ou_epsilon_end=0.01, use_decaying_noise=True):
        self.net = net
        self.device = device
        self.ou_enabled = ou_enabled
        self.ou_mu = ou_mu
        self.ou_teta = ou_teta
        self.ou_sigma = ou_sigma
        self.ou_epsilon = ou_epsilon
        self.ou_decay_steps = ou_decay_steps
        self.ou_epsilon_end = ou_epsilon_end
        self.use_decaying_noise = use_decaying_noise
        self.num_agent_calls = 0

    def initial_state(self):
        return None

    def __call__(self, states, agent_states):
        states_v = float32_preprocessor(states).to(self.device)
        # we use the deterministic output of the actor as the expected value
        mu_v = self.net(states_v)
        actions = mu_v.data.cpu().numpy()

        if self.ou_enabled and self.ou_epsilon > 0:
            new_a_states = []
            for a_state, action in zip(agent_states, actions):
                if a_state is None:
                    # initialization of the OU process
                    a_state = np.zeros(shape=action.shape, dtype=np.float32)
                a_state += self.ou_teta * (self.ou_mu - a_state)
                a_state += self.ou_sigma * np.random.normal(size=action.shape)
                if self.use_decaying_noise:
                    epsilon = max(self.ou_epsilon_end, self.ou_epsilon - self.num_agent_calls / self.ou_decay_steps)
                else:
                    epsilon = self.ou_epsilon
                action += epsilon * a_state
                new_a_states.append(a_state)
        else:
            new_a_states = agent_states

        self.num_agent_calls += 1
        actions = np.clip(actions, -1, 1)
        return actions, new_a_states
