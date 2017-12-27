from src.subproc_vec_env import SubprocVecEnv
from src.atari_wrappers import make_atari, wrap_deepmind

from src.policy import CnnPolicy
from src.a2c import learn

import os

import gym
import argparse
import logging


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    parser.add_argument('--steps', help='training steps', type=int, default=int(1e3))
    parser.add_argument('--nenv', help='No. of environments', type=int, default=2)
    parser.add_argument('--new', help='new session', action='store_true')
    return parser.parse_args()


def train(env_id, num_timesteps, num_cpu, seed=0):
    def make_env(rank):
        def _thunk():
            env = make_atari(env_id)
            env.seed(seed + rank)
            # addinational monitor
            gym.logger.setLevel(logging.WARN)
            return wrap_deepmind(env)
        return _thunk

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    learn(CnnPolicy, env, seed, total_timesteps=int(num_timesteps * 1.1))
    env.close()
    pass


def main():
    args = get_args()
    # Create a model folder
    if not os.path.exists('models'):
        os.mkdir('models')
    train(args.env, args.steps, num_cpu=args.nenv)


if __name__ == "__main__":
    main()
    print(' - - - End of the Program - - - ')
