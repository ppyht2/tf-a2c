from src.subproc_vec_env import SubprocVecEnv
from src.atari_wrappers import make_atari, wrap_deepmind
import gym
import argparse
import logging




def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='CartPole-v0')
    parser.add_argument('--steps', type=int, default=int(1e4))
    return parser.parse_args()



def train(env_id, num_timesteps, num_cpu, seed = 0):
    def thunk_env(rank):
        env = make_atari(env_id)
        env.seed(seed + rank)
        # addinational monitor
        gym.logger.setLevel(logging.WARN)
        return wrap_deepmind(env, clip_rewards = False)
        #env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False):

    # TODO: set global seeds
    # start multiple environments
    env = SubprocVecEnv([ thunk_env(i) for i in range(num_cpu)])
    print('Training . . . ')
    env.close()
    pass

def main():
    args = get_args()
    train(args.env, args.steps, num_cpu=2)

if __name__ == "__main__":
    main()
    print(' - - - End of the Program - - - ')
