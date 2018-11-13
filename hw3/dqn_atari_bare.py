import argparse
from collections import deque

import gym
from atari_wrappers import wrap_deepmind
from replay_memory_atari import ReplayMemory

import random
import numpy as np


def create_model(observation_space, action_space, args):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Discrete)
    assert len(observation_space.shape) == 3

    # TODO: Create a model:
    #   - input to the model is the same as observation_space.shape,
    #     except the last dimension, which should be args.hist_len
    #   - output of the model is action_space.n Q-values
    # Use standard DQN architecture:
    #   1. 8x8 convolution with 32 filters, stride 4, relu activation
    #   2. 4x4 convolution with 64 filters, stride 2, relu activation
    #   3. 3x3 convolution with 64 filters, stride 1, relu activation
    #   4. flatten
    #   5. fully-connected layer with 512 nodes, relu activation
    #   6. fully-connected layer with action_space.n nodes, linear
    # Use mean squared error loss function and RMSprop optimizer.

    # YOUR CODE HERE
    raise NotImplementedError

    return model


def train_model(model, target_model, batch, args):
    obs, actions, rewards, dones, obs_next = batch

    # TODO: Train a model:
    #   0. Normalize the observations (this and next step), you can just divide by 255.0.
    #      This can be also done as part of your network.
    #   1. Do feedforward pass for this observation with main model and for next observation with target model.
    #   2. Calculate Q-value regression targets, cutting the returns at episode end.
    #      if episode end:
    #        Q(s,a) = r(s,a)
    #      else:
    #        Q(s,a) = r(s,a) + gamma * max_a' Q'(s', a')
    #      NB! You are working with vectors, so try to do away without for loops!
    #   3. Set the targets for Q-values of chosen actions.
    #      NB! For other Q-values you might need to set their targets the same as their original values, so the error is zero.
    #   4. Train the main model with one gradient update. Return loss and mean Q-value for statistics.
    #   5. Implement Double Q-learning:
    #      Q(s,a) = r(s,a) + gamma * Q'(s', argmax_a' Q(s', a'))
    #      NB! You need to do additional forward pass for this!

    # YOUR CODE HERE
    raise NotImplementedError

    return loss, qmean


def update_target(model, target_model):
    # TODO: Copy main model weights to target model.

    # YOUR CODE HERE
    raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='PongNoFrameskip-v4')
    parser.add_argument('--exp_name', type=str, default='pong')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--num_timesteps', type=int, default=5000000)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--hist_len', type=int, default=4)
    parser.add_argument('--epsilon_start', type=float, default=1.0)
    parser.add_argument('--epsilon_end', type=float, default=0.1)
    parser.add_argument('--epsilon_steps', type=int, default=1000000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--learning_start', type=int, default=50000)
    parser.add_argument('--update_freq', type=int, default=10000)
    parser.add_argument('--train_freq', type=int, default=4)
    parser.add_argument('--save_freq', type=int, default=100000)
    parser.add_argument('--log_freq', type=int, default=10000)
    args = parser.parse_args()

    # create environment and add standard wrappers
    env = gym.make(args.env_name)
    env = wrap_deepmind(env)

    # create main model and target model.
    model = create_model(env.observation_space, env.action_space, args)
    target_model = create_model(env.observation_space, env.action_space, args)
    # copy main model weights to target
    update_target(model, target_model)

    # create replay memory
    replay_memory = ReplayMemory(args.replay_size, env.observation_space.shape, args.hist_len)

    # statistics
    loss = 0
    qmean = 0
    rewards = []
    lengths = []
    episode_num = 0
    episode_reward = 0
    episode_length = 0
    num_iterations = 0

    # reset the environment
    obs = env.reset()
    # create fifo queue of last arg.hist_len observations
    obs_hist = deque([obs] * args.hist_len, maxlen=args.hist_len)
    # loop for args.num_timesteps steps
    for t in range(args.num_timesteps):
        # calculate exploration rate
        if t < args.epsilon_steps:
            # anneal epsilon linearly from args.epsilon_start to args.epsilon_end
            epsilon = args.epsilon_start - (t / args.epsilon_steps) * (args.epsilon_start - args.epsilon_end)
        else:
            # after ars.epsilon_steps steps set exploration to fixed rate
            epsilon = args.epsilon_end

        # TODO: Choose action:
        #  - with probability epsilon choose random action,
        #  - otherwise choose action with maximum Q-value.

        # YOUR CODE HERE
        raise NotImplementedError

        # step environment
        next_obs, reward, done, info = env.step(action)
        if args.render:
            env.render()

        # TODO: Add current experience to replay memory

        # YOUR CODE HERE
        raise NotImplementedError

        # statistics
        episode_reward += reward
        episode_length += 1
        # if episode ended
        if done:
            # reset environment
            obs = env.reset()
            # fill history with the first observation
            obs_hist.extend([obs] * args.hist_len)

            # statistics
            episode_num += 1
            rewards.append(episode_reward)
            lengths.append(episode_length)
            episode_reward = 0
            episode_length = 0
        else:
            # otherwise add new observation to history
            obs = next_obs
            obs_hist.append(obs)

        # training
        if t >= args.learning_start and t % args.train_freq == 0:
            # TODO: Train model:
            #  - sample minibatch from replay memory
            #  - train model with that minibatch

            # YOUR CODE HERE
            raise NotImplementedError

            num_iterations += 1

            # update target model after fixed number of iterations
            if num_iterations % args.update_freq == 0:
                update_target(model, target_model)

            # save model after fixed number of iterations
            if num_iterations % args.save_freq == 0:
                # TODO: save model
                pass

        # log statistics
        if (t + 1) % args.log_freq == 0:
            print("Timestep %d: episodes %d, reward %f, length %f, iterations: %d, epsilon: %f, loss: %f, Q-mean: %f, replay: %d" %
                  (t + 1, len(rewards), np.mean(rewards), np.mean(lengths), num_iterations, epsilon, loss, qmean, replay_memory.count))
            rewards = []
            lengths = []

    print("Done")
