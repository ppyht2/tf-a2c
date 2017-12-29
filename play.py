import argparse
import os
import numpy as np
from src.atari_wrappers import make_atari, wrap_deepmind
from src.a2c import Model
from src.policy import CnnPolicy


def get_args():
    # Get some basic command line arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', help='environment ID', default='BreakoutNoFrameskip-v4')
    return parser.parse_args()


def get_model(env, nsteps=5, nstack=4, total_timesteps=int(80e6),
              vf_coef=0.5, ent_coef=0.01, max_grad_norm=0.5, lr=7e-4,
              lrschedule='linear', epsilon=1e-5, alpha=0.99, gamma=0.99, log_interval=100):

    ob_space = env.observation_space
    ac_space = env.action_space
    # Load model
    model = Model(policy=CnnPolicy, ob_space=ob_space, ac_space=ac_space, nenvs=1,
                  nsteps=nsteps, nstack=1, num_procs=1,
                  ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm,
                  lr=lr, alpha=alpha, epsilon=epsilon, total_timesteps=total_timesteps,
                  lrschedule=lrschedule)

    return model


def main():
    env_id = get_args().env
    env = make_atari(env_id)
    env = wrap_deepmind(env, frame_stack=True, clip_rewards=False, episode_life=True)

    model = get_model(env)

    # check for save path
    save_path = os.path.join('models', env_id + '.save')
    model.load(save_path)

    obs = env.reset()
    while True:
        obs = np.expand_dims(obs.__array__(), axis=0)
        a, v, _ = model.step(obs)
        obs, reward, done, info = env.step(a)
        env.render()
        if done:
            env.reset()


if __name__ == '__main__':
    main()
