import gym
import time
from atari_wrappers import wrap_deepmind

env = gym.make('PongNoFrameskip-v4')
#env = gym.wrappers.Monitor(env, "gym", video_callable=False, force=True)
env = wrap_deepmind(env)

print(env.observation_space.shape)

start = time.time()
obs = env.reset()
for i in range(50000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        env.reset()
print("Duration:", time.time() - start)
