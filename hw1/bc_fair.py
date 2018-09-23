import argparse
import pickle
import os
import csv
import gym
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from load_policy import load_policy
import tf_util
import tensorflow as tf


def create_model(input_size, output_size, args):
    h = x = Input(shape=(input_size,))
    for i in range(args.hidden_layers):
        h = Dense(args.hidden_nodes, activation=args.activation_function)(h)
    y = Dense(output_size)(h)

    model = Model(x, y)
    model.compile(optimizer='adam', loss='mse')
    return model


def train_model(model, observations, actions, epoch, args):
    model.fit(observations, actions,
              epochs=epoch + 1,
              initial_epoch=epoch,
              batch_size=args.batch_size)


def eval_model(model, env, obsmean, obsstd, args):
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            obsnorm = (obs - obsmean) / (obsstd + np.finfo(float).eps)
            action = model.predict(obsnorm[np.newaxis, :])
            observations.append(obs)
            actions.append(action[0])
            obs, r, done, _ = env.step(action[0])
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)
    return returns, observations, actions


def collect_data(policy_fn, env, args):
    max_steps = args.max_timesteps or env.spec.timestep_limit

    returns = []
    observations = []
    actions = []
    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_fn(obs[np.newaxis, :])
            observations.append(obs)
            actions.append(action[0])
            obs, r, done, _ = env.step(action[0])
            totalr += r
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break
        returns.append(totalr)
    return returns, observations, actions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('envname', type=str)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--max_timesteps', type=int)
    parser.add_argument('--num_rollouts', type=int, default=10,
                        help='Number of expert roll outs')
    # model parameters
    parser.add_argument('--hidden_layers', type=int, default=2)
    parser.add_argument('--hidden_nodes', type=int, default=64)
    parser.add_argument('--activation_function', choices=['sigmoid', 'tanh', 'relu'], default='tanh')
    # training parameters
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # load dataset
    with open(os.path.join('expert_data', args.envname + '.pkl'), 'rb') as f:
        data = pickle.load(f)
    observations = data['observations']
    actions = data['actions'][:, 0, :]

    print("Environment:", args.envname)
    print("Observations:", observations.shape)
    print("Actions:", actions.shape)

    # load expert policy
    policy_fn = load_policy(os.path.join('experts', args.envname + '.pkl'))

    # create environment
    env = gym.make(args.envname)
    assert env.observation_space.shape == observations.shape[1:]
    assert env.action_space.shape == actions.shape[1:]

    # normalize observations
    obsmean = np.mean(observations, axis=0)
    obsstd = np.std(observations, axis=0)
    observations = (observations - obsmean) / (obsstd + np.finfo(float).eps)

    # open results file
    csvfile = open(os.path.join('bc_results', args.envname + '.csv'), 'w')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Reward mean", "Reward std"])

    # main training loop
    with tf.Session():
        tf_util.initialize()
        model = create_model(observations.shape[1], actions.shape[1], args)
        for epoch in range(args.num_epochs):
            train_model(model, observations, actions, epoch, args)
            print("Epoch %d evaluation:" % (epoch + 1), end=' ', flush=True)
            returns, _, _ = eval_model(model, env, obsmean, obsstd, args)
            print("reward mean %f±%f" % (np.mean(returns), np.std(returns)))
            csvwriter.writerow([epoch + 1, np.mean(returns), np.std(returns)])

            # collect additional data to be comparable to DAgger
            _, new_obs, new_act = collect_data(policy_fn, env, args)
            # normalize new observations
            new_obs = (new_obs - obsmean) / (obsstd + np.finfo(float).eps)
            # add new observations and actions to the dataset
            observations = np.append(observations, new_obs, axis=0)
            actions = np.append(actions, new_act, axis=0)

    csvfile.close()
    print("Done")
