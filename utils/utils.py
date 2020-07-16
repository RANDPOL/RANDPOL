"""
Utility functions.
Most taken from the PTAN (PyTorch Agent Net) library by Shmuma
https://github.com/Shmuma/ptan 
and https://github.com/taldatech/pytorch-ls-ddpg
"""

# imports
import numpy as np
import torch
import time
import sys
import collections
import os
from utils.nn_agent_models import float32_preprocessor


def test_net(net, env, agent_model, count=10, device="cpu"):
    rewards = 0.0
    steps = 0
    for _ in range(count):
        obs = env.reset()
        while True:
            obs_v = agent_model.float32_preprocessor([obs]).to(device)
            mu_v = net(obs_v)
            action = mu_v.squeeze(dim=0).data.cpu().numpy()
            action = np.clip(action, -1, 1)
            obs, reward, done, _ = env.step(action)
            rewards += reward
            steps += 1
            if done:
                break
    return rewards / count, steps / count


def save_agent_state(act_net, crt_net, optimizers, frame, games, save_replay=False, replay_buffer=None, name='',
                     path=None):
    """
    This function saves the current state of the NN (the weights) to a local file.
    :param act_net: the current actor NN (nn.Module)
    :param crt_net: the current critic NN (nn.Module)
    :param optimizers: the network's optimizer (torch.optim)
    :param frame: current frame number (int)
    :param games: total number of games seen (int)
    :param save_replay: whether or not to save the replay buffer (bool)
    :param replay_buffer: the replay buffer (list)
    :param name: specific name for the checkpoint (str)
    :param path: path to specific location where to save (str)
    """
    dir_name = './agent_ckpt'
    if path:
        full_path = path
    else:
        if name:
            filename = "agent_randpol_" + name + ".pth"
        else:
            filename = "agent_randpol.pth"
        dir_name = './agent_ckpt'
        full_path = os.path.join(dir_name, filename)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    if save_replay and replay_buffer is not None:
        torch.save({
            'act_state_dict': act_net.state_dict(),
            'crt_state_dict': crt_net.state_dict(),
            'act_optimizer_state_dict': optimizers[0].state_dict(),
            'crt_optimizer_state_dict': optimizers[1].state_dict(),
            'frame_count': frame,
            'games': games,
            'replay_buffer': replay_buffer
        }, full_path)
    else:
        torch.save({
            'act_state_dict': act_net.state_dict(),
            'crt_state_dict': crt_net.state_dict(),
            'act_optimizer_state_dict': optimizers[0].state_dict(),
            'crt_optimizer_state_dict': optimizers[1].state_dict(),
            'frame_count': frame,
            'games': games
        }, full_path)
    print("Saved Agent checkpoint @ ", full_path)


def load_agent_state(act_net, crt_net, optimizers, path=None, copy_to_target_network=False, load_optimizer=True,
                     target_nets=None, buffer=None, load_buffer=False):
    """
    This function loads a state of the NN (the weights) from a local file.
    :param act_net: the current actor NN (nn.Module)
    :param crt_net: the current critic NN (nn.Module)
    :param optimizers: the network's optimizers (torch.optim)
    :param path: full path to checkpoint file (.pth) (str)
    :param copy_to_target_network: whether or not to copy the weights to target network (bool)
    :param load_optimizer: whether or not to load the optimizer state (bool)
    :param load_buffer: whether or not to load the replay buffer (bool)
    :param buffer: the replay buffer
    :param target_nets: the target NNs
    """
    if path is None:
        raise SystemExit("path to model must be specified")
    else:
        full_path = path
    exists = os.path.isfile(full_path)
    if exists:
        if not torch.cuda.is_available():
            checkpoint = torch.load(full_path, map_location='cpu')
        else:
            checkpoint = torch.load(full_path)
        act_net.load_state_dict(checkpoint['act_state_dict'])
        crt_net.load_state_dict(checkpoint['crt_state_dict'])
        if load_optimizer:
            optimizers[0].load_state_dict(checkpoint['act_optimizer_state_dict'])
            optimizers[1].load_state_dict(checkpoint['crt_optimizer_state_dict'])
        # self.steps_count = checkpoint['steps_count']
        # self.episodes_seen = checkpoint['episodes_seen']
        # selector.epsilon = checkpoint['epsilon']
        # self.num_param_update = checkpoint['num_param_updates']
        print("Checkpoint loaded successfully from ", full_path)
        # # for manual loading a checkpoint
        if copy_to_target_network and target_nets is not None:
            target_nets[0].sync()
            target_nets[1].sync()
        if load_buffer and buffer is not None:
            buffer.buffer = checkpoint['replay_buffer']
    else:
        print("No checkpoint found...")


def unpack_batch(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = float32_preprocessor(states).to(device)
    actions_v = float32_preprocessor(actions).to(device)
    rewards_v = float32_preprocessor(rewards).to(device)
    last_states_v = float32_preprocessor(last_states).to(device)
    dones_t = torch.ByteTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v


class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB

    Designed and tested with pytorch-tensorboard in mind
    """
    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            if torch.is_tensor(data[0]):
                self.writer.add_scalar(param_name, torch.mean(torch.stack(data)), iter_index)
            else:
                self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()


class RewardTracker:
    def __init__(self, writer, min_ts_diff=1.0):
        """
        Constructs RewardTracker
        :param writer: writer to use for writing stats
        :param min_ts_diff: minimal time difference to track speed
        """
        self.writer = writer
        self.min_ts_diff = min_ts_diff

    def __enter__(self):
        self.ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        mean_reward = np.mean(self.total_rewards[-100:])
        ts_diff = time.time() - self.ts
        if ts_diff > self.min_ts_diff:
            speed = (frame - self.ts_frame) / ts_diff
            self.ts_frame = frame
            self.ts = time.time()
            epsilon_str = "" if epsilon is None else ", eps %.2f" % epsilon
            print("Steps:%d   episodes:%d, mean reward %.3f" % (
                frame, len(self.total_rewards), mean_reward
            ))
            sys.stdout.flush()
            self.writer.add_scalar("speed", speed, frame)
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        return mean_reward if len(self.total_rewards) > 30 else None

