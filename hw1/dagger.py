import argparse
import pickle
import os
import csv
import gym
import numpy as np

from bc import create_model, train_model, eval_model
from load_policy import load_policy
import tf_util
import tensorflow as tf

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
    csvfile = open(os.path.join('dagger_results', args.envname + '.csv'), 'w')
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(["Epoch", "Reward mean", "Reward std"])

    # main training loop
    with tf.Session():
        tf_util.initialize()
        model = create_model(observations.shape[1], actions.shape[1], args)
        for epoch in range(args.num_epochs):
            train_model(model, observations, actions, epoch, args)
            print("Epoch %d evaluation:" % (epoch + 1), end=' ', flush=True)
            returns, new_obs, _ = eval_model(model, env, obsmean, obsstd, args)
            print("reward mean %f±%f" % (np.mean(returns), np.std(returns)))
            csvwriter.writerow([epoch + 1, np.mean(returns), np.std(returns)])

            # compute expert actions for observations
            new_act = policy_fn(new_obs)
            # normalize new observations
            new_obs = (new_obs - obsmean) / (obsstd + np.finfo(float).eps)
            # add new observations and actions to the dataset
            observations = np.append(observations, new_obs, axis=0)
            actions = np.append(actions, new_act, axis=0)

    csvfile.close()
    print("Done")
