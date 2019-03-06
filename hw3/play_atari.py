import gym
import numpy as np
from collections import deque
from keras.models import load_model
from atari_wrappers import wrap_deepmind
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('model_name', type=str)
    parser.add_argument('--hist_len', type=int, default=4)
    parser.add_argument('--num_episodes', type=int, default=10)
    args = parser.parse_args()

    model = load_model(args.model_name)
    env = gym.make(args.env_name)
    env = wrap_deepmind(env)

    for i in range(args.num_episodes):
        obs = env.reset()
        obs_hist = deque([obs] * args.hist_len, maxlen=args.hist_len)
        done = False
        while not done:
            obs_input = np.concatenate(obs_hist, axis=2)
            action = model.predict(obs_input[np.newaxis])
            action = np.argmax(action[0])

            obs, reward, done, _ = env.step(action)
            env.render()
            obs_hist.append(obs)

        time.sleep(0.5)
