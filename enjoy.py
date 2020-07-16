"""
Play a trained agent
"""


import gym
import pybullet_envs
import numpy as np
import utils.nn_agent_models as agent_model
import torch
import random
import argparse
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="playing RANDPOL Agent")
    parser.add_argument("-e", "--env", type=str,
                        help="environment to play", default="MinitaurBulletEnv-v0")
    parser.add_argument("-y", "--path", type=str, help="path to agent to play")

    parser.add_argument("-x", "--record", help="directory to save rendering")
    parser.add_argument("--no-visualize", default=True, action='store_false', dest='visualize',
                        help="Disable visualization")

    args = parser.parse_args()


    if args.path:
        path_to_model_ckpt = args.path
    else:
        raise SystemExit("no path given")
    render = True
    spec = gym.envs.registry.spec(args.env)
    if spec._kwargs.get('render') and render:
        spec._kwargs['render'] = True
    env = gym.make(args.env)
    use_constant_seed = True
    seed = 1234
    if use_constant_seed:
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)
        torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print("Random seed:", seed)
    if args.record:

        env = gym.wrappers.Monitor(env, args.record)

    net = agent_model.Actor(env.observation_space.shape[0], env.action_space.shape[0])
    net.load_state_dict(torch.load(path_to_model_ckpt))

    obs = env.reset()
    total_reward = 0.0
    total_steps = 0
    while True:
        obs_v = torch.FloatTensor([obs])
        mu_v = net(obs_v)
        action = mu_v.squeeze(dim=0).data.numpy()
        action = np.clip(action, -1, 1)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        total_steps += 1
        if render:
            env.render()
        if done:
            env.close()
            break
    print("Ran %d steps we got %.3f reward" % (total_steps, total_reward))
