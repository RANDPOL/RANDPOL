"""
Train RANDPOL Agent
"""

import os
import time
import gym
import pybullet_envs
from tensorboardX import SummaryWriter
import numpy as np
import utils.nn_agent_models as agent_model
import utils.Experience as Experience
import utils.utils as utils

import torch
import torch.optim as optim
import torch.nn.functional as F
import random
from utils.utils import test_net
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)


REWARD_TO_SOLVE = None  # mean reward the environment is considered SOLVED, None if not known.

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="training RANDPOL Agent")

	parser.add_argument("-n", "--name", type=str,
						help="model name, for saving and loading,"
							 " if not set, training will continue from a pretrained checkpoint", default = "RANDPOL")
	parser.add_argument("-e", "--env", type=str,
						help="environment", default="MinitaurBulletEnv-v0")

	parser.add_argument("-d", "--decay_rate", type=int,
						help="number of episodes for epsilon decaying, default: 500000", default= 500000)
	parser.add_argument("-o", "--optimizer", type=str,
						help="optimizing algorithm ('RMSprop', 'Adam'), default: 'Adam'", default = 'Adam')
	parser.add_argument("--lr_actor", type=float,
						help="learning rate for the Actor optimizer, default: 0.0001", default = 0.0001)
	parser.add_argument("--lr_critic", type=float,
						help="learning rate for the Critic optimizer, default: 0.0001", default = 0.0001)

	parser.add_argument("-g", "--gamma", type=float,
						help="discount factor, default: 0.99", default = 0.99)
	parser.add_argument("-s", "--buffer_size", type=int,
						help="Replay Buffer size, default: 1000000", default = 1000000)

	parser.add_argument("-b", "--batch_size", type=int,
						help="number of samples in each batch, default: 64", default = 64)
	parser.add_argument("-i", "--steps_to_start_learn", type=int,
						help="number of random steps before the agents starts learning, default: 10000", default = 100)
	parser.add_argument("-c", "--test_iter", type=int,
						help="number of iterations between policy testing, default: 10000", default = 10000)

	args = parser.parse_args()
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



	# Training


	model_name = args.name
	decay_rate = args.decay_rate
	lr_actor = args.lr_actor
	lr_critic = args.lr_critic
	gamma = args.gamma
	replay_size = args.buffer_size
	batch_size = args.batch_size
	steps_to_start_learn = args.steps_to_start_learn
	test_iter = args.test_iter

	model_saving_path = './agent_ckpt/agent_' + model_name + ".pth"

	env = gym.make(args.env)
	test_env = gym.make(args.env)
	name = model_name + "_agent_" + args.env


	name += "-BATCH-" + str(batch_size)
	save_path = os.path.join("saves", "RANDPOL-" + name)
	os.makedirs(save_path, exist_ok=True)
	ckpt_save_path = './agent_ckpt/' + name + ".pth"
	if not os.path.exists('./agent_ckpt/'):
		os.makedirs('./agent_ckpt')

	training_random_seed = 123
		
	
	name += "-SEED-" + str(training_random_seed)
	np.random.seed(training_random_seed)
	random.seed(training_random_seed)
	env.seed(training_random_seed)
	test_env.seed(training_random_seed)
	torch.manual_seed(training_random_seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(training_random_seed)
	

	act_net = agent_model.Actor(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
	crt_net = agent_model.Critic(env.observation_space.shape[0], env.action_space.shape[0]).to(device)
	print(act_net)
	print(crt_net)
	tgt_act_net = agent_model.TargetNet(act_net)
	tgt_crt_net = agent_model.TargetNet(crt_net)

	writer = SummaryWriter(comment="-RANDPOL-" + name)
	if decay_rate is not None:
		agent = agent_model.AgentRANDPOL(act_net, device=device, ou_decay_steps=decay_rate)
	else:
		agent = agent_model.AgentRANDPOL(act_net, device=device)
	exp_source = Experience.ExperienceSourceFirstLast(env, agent, gamma=gamma, steps_count=1)
	buffer = Experience.ExperienceReplayBuffer(exp_source, buffer_size=replay_size)
	if args.optimizer and args.optimizer == "RMSprop":
		act_opt = optim.RMSprop(filter(lambda p: p.requires_grad, act_net.parameters()), lr=lr_actor)
		crt_opt = optim.RMSprop(filter(lambda p: p.requires_grad, crt_net.parameters()), lr=lr_critic)
	else:
		act_opt = optim.Adam(filter(lambda p: p.requires_grad, act_net.parameters()), lr=lr_actor)
		crt_opt = optim.Adam(filter(lambda p: p.requires_grad, crt_net.parameters()), lr=lr_critic)

	utils.load_agent_state(act_net, crt_net, [act_opt, crt_opt], path=ckpt_save_path)

	frame_idx = 0
	drl_updates = 0
	best_reward = None
	with utils.RewardTracker(writer) as tracker:
		with utils.TBMeanTracker(writer, batch_size=10) as tb_tracker:
			while True:
				frame_idx += 1
				buffer.populate(1)
				rewards_steps = exp_source.pop_rewards_steps()
				if rewards_steps:
					rewards, steps = zip(*rewards_steps)
					tb_tracker.track("episode_steps", steps[0], frame_idx)
					mean_reward = tracker.reward(rewards[0], frame_idx)
					if mean_reward is not None and REWARD_TO_SOLVE is not None and mean_reward > REWARD_TO_SOLVE:
						print("environment solved in % steps" % frame_idx,
							  " (% episodes)" % len(tracker.total_rewards))
						utils.save_agent_state(act_net, crt_net, [act_opt, crt_opt], frame_idx,
											   len(tracker.total_rewards), path=ckpt_save_path)
						break

				if len(buffer) < steps_to_start_learn:
					continue

				batch = buffer.sample(batch_size)
				states_v, actions_v, rewards_v, dones_mask, last_states_v = utils.unpack_batch(batch, device)

				# train critic
				crt_opt.zero_grad()
				q_v = crt_net(states_v, actions_v)
				last_act_v = tgt_act_net.target_model(last_states_v)
				q_last_v = tgt_crt_net.target_model(last_states_v, last_act_v)
				q_last_v[dones_mask] = 0.0
				q_ref_v = rewards_v.unsqueeze(dim=-1) + q_last_v * gamma
				critic_loss_v = F.mse_loss(q_v, q_ref_v.detach())
				critic_loss_v.backward()
				crt_opt.step()
				tb_tracker.track("loss_critic", critic_loss_v, frame_idx)
				tb_tracker.track("critic_ref", q_ref_v.mean(), frame_idx)



				# train actor
				act_opt.zero_grad()
				cur_actions_v = act_net(states_v)
				actor_loss_v = -crt_net(states_v, cur_actions_v)
				actor_loss_v = actor_loss_v.mean()
				actor_loss_v.backward()
				act_opt.step()
				tb_tracker.track("loss_actor", actor_loss_v, frame_idx)

				tgt_act_net.alpha_sync(alpha=1 - 1e-3)
				tgt_crt_net.alpha_sync(alpha=1 - 1e-3)

				if frame_idx % test_iter == 0:
					ts = time.time()
					rewards, steps = test_net(act_net, test_env, agent_model, device=device)
					print("Test done in %.2f sec, reward %.3f, steps %d" % (
						time.time() - ts, rewards, steps))
					writer.add_scalar("test_reward", rewards, frame_idx)
					writer.add_scalar("test_steps", steps, frame_idx)
					if best_reward is None or best_reward < rewards:
						if best_reward is not None:
							print("Best reward updated: %.3f -> %.3f" % (best_reward, rewards))
							name = "RANDPOL_reward_%+.3f_%d.dat" % (rewards, frame_idx)
							fname = os.path.join(save_path, name)
							torch.save(act_net.state_dict(), fname)
							utils.save_agent_state(act_net, crt_net, [act_opt, crt_opt], frame_idx,
												   len(tracker.total_rewards), path=ckpt_save_path)
						best_reward = rewards
