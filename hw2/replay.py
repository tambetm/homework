import gym
import numpy as np
from keras.models import load_model
import time
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('model_name', type=str)
    parser.add_argument('--num_episodes', type=int, default=10)
    args = parser.parse_args()

    model = load_model(args.model_name)
    env = gym.make(args.env_name)

    for i in range(args.num_episodes):
        obs = env.reset()
        done = False
        while not done:
            action = model.predict(obs[np.newaxis, :])
            if len(action) == 2:
                action = action[0]

            if isinstance(env.action_space, gym.spaces.Discrete):
                action = np.argmax(action[0])
            else:
                action = np.reshape(action[0], env.action_space.shape)

            obs, reward, done, _ = env.step(action)
            env.render()

        time.sleep(0.5)
