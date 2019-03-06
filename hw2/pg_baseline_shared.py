import argparse
import os
import csv
import json
import time
import gym
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import RMSprop


def create_model(observation_space, action_space, args):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box) \
        or isinstance(action_space, gym.spaces.Discrete)

    h = x = Input(shape=observation_space.shape)
    for i in range(args.hidden_layers):
        h = Dense(args.hidden_nodes, activation=args.activation_function)(h)

    # baseline output
    bh = x
    for i in range(args.hidden_layers):
        bh = Dense(args.hidden_nodes, activation=args.activation_function)(bh)
    b = Dense(1)(bh)

    if isinstance(action_space, gym.spaces.Discrete):
        y = Dense(action_space.n, activation='softmax')(h)
        model = Model(x, [y, b])
        model.compile(optimizer=RMSprop(lr=args.learning_rate), loss=['sparse_categorical_crossentropy', 'mse'], loss_weights=[1, args.baseline_weight])
    else:
        y = Dense(np.prod(action_space.shape))(h)
        # TODO: learnable stddev?
        model = Model(x, [y, b])
        model.compile(optimizer=RMSprop(lr=args.learning_rate), loss=['mse', 'mse'], loss_weights=[1, args.baseline_weight])

    model.summary()
    return model


def train_model(model, observations, actions, returns, advantages, args):
    # prevent divide by zero in Keras
    advantages += np.finfo(float).eps
    # normalize baseline targets
    returns = (returns - np.mean(returns)) / (np.std(returns) + np.finfo(float).eps)
    return model.train_on_batch(observations, [actions, returns], sample_weight=[advantages, None])


def sample_trajectories(env, model, args):
    max_steps = args.max_timesteps or env.spec.timestep_limit

    observations = []
    actions = []
    rewards = []
    baselines = []
    total_steps = 0
    while total_steps < args.batch_size:
        observations.append([])
        actions.append([])
        rewards.append([])
        baselines.append([])

        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action, baseline = model.predict(obs[np.newaxis, :])
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = np.random.choice(env.action_space.n, p=action[0])
            else:
                action = np.reshape(action[0], env.action_space.shape)
                # TODO: implement learnable stddev
                action = np.random.normal(action, scale=args.stddev)

            observations[-1].append(obs)
            actions[-1].append(action)
            baselines[-1].append(baseline[0, 0])
            obs, reward, done, _ = env.step(action)
            rewards[-1].append(reward)

            steps += 1
            if args.render and total_steps == 0:
                env.render()
            if steps >= max_steps:
                break

        total_steps += steps

    return observations, actions, rewards, baselines


def compute_returns(rewards, args):
    if args.reward_to_go:
        returns = []
        for eps_rew in rewards:
            eps_ret = []
            ret = 0
            for rew in reversed(eps_rew):
                ret = rew + args.discount * ret
                eps_ret.insert(0, ret)
            returns.append(eps_ret)
    else:
        returns = []
        for eps_rew in rewards:
            ret = 0
            for rew in reversed(eps_rew):
                ret = rew + args.discount * ret
            eps_ret = [ret] * len(eps_rew)
            returns.append(eps_ret)

    return returns


def compute_advantages(returns, baselines, args):
    # rescale baselines to be of the same magnitude as returns
    baselines = baselines * np.std(returns) + np.mean(returns)
    advantages = returns - baselines

    if not args.dont_normalize_advantages:
        advmean = np.mean(advantages)
        advstd = np.std(advantages)
        advantages = (advantages - advmean) / (advstd + np.finfo(float).eps)

    return advantages


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--max_timesteps', '-ep', type=float)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--stddev', type=float, default=0.1)
    parser.add_argument('--baseline_weight', type=float, default=1.)
    #parser.add_argument('--nn_baseline', '-bl', action='store_true')
    #parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--hidden_layers', '-l', type=int, default=2)
    parser.add_argument('--hidden_nodes', '-s', type=int, default=64)
    parser.add_argument('--activation_function', choices=['sigmoid', 'tanh', 'relu'], default='tanh')
    args = parser.parse_args()

    # create environment
    print("Environment:", args.env_name)
    env = gym.make(args.env_name)
    print("Observations:", env.observation_space)
    print("Actions:", env.action_space)

    # loop over experiments
    for e in range(args.n_experiments):
        # create model
        model = create_model(env.observation_space, env.action_space, args)

        # create experiment directory
        logdir = os.path.join('data', args.exp_name + '_' + args.env_name, str(e))
        if not os.path.exists(logdir):
            os.makedirs(logdir)

        # open results file
        csvfile = open(os.path.join(logdir, 'log.txt'), 'w')
        csvwriter = csv.writer(csvfile, delimiter='\t')
        csvwriter.writerow(["Time", "Iteration", "AverageReturn", "StdReturn", "MaxReturn", "MinReturn",
                           "EpLenMean", "EpLenStd", "TimestepsThisBatch", "TimestepsSoFar"])

        # write params file
        with open(os.path.join(logdir, 'params.json'), 'w') as f:
            json.dump(vars(args), f)

        # main training loop
        total_timesteps = 0
        start = time.time()
        for i in range(args.n_iter):
            # sample trajectories
            observations, actions, rewards, baselines = sample_trajectories(env, model, args)
            # compute returns
            returns = compute_returns(rewards, args)
            # flatten returns and baselines
            returns = np.concatenate(returns)
            baselines = np.concatenate(baselines)
            # compute advantages
            advantages = compute_advantages(returns, baselines, args)
            # flatten observations and actions
            observations = np.concatenate(observations)
            actions = np.concatenate(actions)
            # train model
            losses = train_model(model, observations, actions, returns, advantages, args)

            # log statistics
            returns = [sum(eps_rew) for eps_rew in rewards]
            lengths = [len(eps_rew) for eps_rew in rewards]
            total_timesteps += sum(lengths)
            print("Iteration %d:" % (i + 1),
                  "reward mean %f±%f" % (np.mean(returns), np.std(returns)),
                  "episode length %f±%f" % (np.mean(lengths), np.std(lengths)),
                  "total timesteps", total_timesteps,
                  "losses", losses)
            csvwriter.writerow([time.time() - start, i,
                               np.mean(returns), np.std(returns),
                               np.max(returns), np.min(returns),
                               np.mean(lengths), np.std(lengths),
                               sum(lengths), total_timesteps])

        csvfile.close()
        model.save(os.path.join(logdir, 'model.hdf5'))

    print("Done")
