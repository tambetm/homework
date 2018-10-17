import argparse
import os
import csv
import json
import time
import gym
import numpy as np
from keras.layers import Input, Dense, Activation, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.losses import sparse_categorical_crossentropy
from keras.initializers import Constant
from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class SampleGaussian(Layer):
    def __init__(self, initial_std, **kwargs):
        self.initial_std = initial_std
        super(SampleGaussian, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        logstd = self.add_weight(name='logstd',
                                 shape=(1,) + input_shape[1:],
                                 initializer=Constant(np.log(self.initial_std)),
                                 trainable=True)
        # make sure the stddev is positive
        self.std = K.exp(logstd)
        super(SampleGaussian, self).build(input_shape)

    def call(self, x):
        return x + self.std * K.random_normal(K.shape(x))


def create_model(observation_space, action_space, args):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box) \
        or isinstance(action_space, gym.spaces.Discrete)

    h = x = Input(shape=observation_space.shape)
    for i in range(args.hidden_layers):
        h = Dense(args.hidden_nodes, activation=args.activation_function)(h)

    if isinstance(action_space, gym.spaces.Discrete):
        # produce logits for all actions
        h = Dense(action_space.n)(h)
        # sample action from logits
        a = Lambda(lambda x: tf.multinomial(x, num_samples=1))(h)
        # turn logits into probabilities
        p = Activation('softmax')(h)
        # model outputs sampled action
        model = Model(x, a)
        # loss is between true values and probabilities
        model.compile(optimizer=Adam(lr=args.learning_rate), loss=lambda y_true, y_pred: sparse_categorical_crossentropy(y_true, p))
    else:
        # number of actions
        n = np.prod(action_space.shape)
        # produce means and stddevs for Gaussian
        mu = Dense(n)(h)
        # sample action from Gaussian
        gaussian = SampleGaussian(initial_std=args.stddev)
        a = gaussian(mu)
        # model outputs sampled action
        model = Model(x, a)
        # log loss of Gaussian probability
        model.compile(optimizer=Adam(lr=args.learning_rate, clipnorm=1.),
                      loss=lambda y_true, y_pred: 0.5 * K.log(2 * np.pi * gaussian.std**2) + 0.5 * (y_true - mu)**2 / gaussian.std**2)

    model.summary()
    return model


def train_model(model, observations, actions, advantages, args):
    # skip training if all advantages are 0s to prevent Keras bug?
    if np.any(advantages):
        # flatten observations and actions
        observations = np.concatenate(observations)
        actions = np.concatenate(actions)
        # train the model using advantages as sample weights
        model.train_on_batch(observations, actions, sample_weight=advantages)


def sample_trajectories(env, model, args):
    max_steps = args.max_timesteps or env.spec.timestep_limit

    observations = []
    actions = []
    rewards = []
    total_steps = 0
    while total_steps < args.batch_size:
        observations.append([])
        actions.append([])
        rewards.append([])

        obs = env.reset()
        done = False
        steps = 0
        while not done:
            action = model.predict(obs[np.newaxis, :])
            if isinstance(env.action_space, gym.spaces.Discrete):
                action = action[0, 0]
            else:
                action = np.reshape(action[0], env.action_space.shape)

            observations[-1].append(obs)
            actions[-1].append(action)
            obs, reward, done, _ = env.step(action)
            rewards[-1].append(reward)

            steps += 1
            if args.render and total_steps == 0:
                env.render()
            if steps >= max_steps:
                break

        total_steps += steps

    return observations, actions, rewards


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


def compute_advantages(returns, args):
    if args.reward_to_go:
        # calculate mean return per timestep
        maxsteps = max([len(eps_ret) for eps_ret in returns])
        retsums = np.zeros(maxsteps)
        retcounts = np.zeros(maxsteps)
        for eps_ret in returns:
            retsums[:len(eps_ret)] += eps_ret
            retcounts[:len(eps_ret)] += 1
        baselines = retsums / retcounts

        # calculate advantage per timestep
        advantages = np.zeros(0)
        for eps_ret in returns:
            advantages = np.append(advantages, eps_ret - baselines[:len(eps_ret)])
    else:
        # calculate mean over episode total rewards
        baseline = np.mean([ret[0] for ret in returns])
        returns = np.concatenate(returns)
        advantages = returns - baseline

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
    parser.add_argument('--stddev', type=float, default=1.)
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
            observations, actions, rewards = sample_trajectories(env, model, args)
            # compute returns
            returns = compute_returns(rewards, args)
            # compute advantages
            advantages = compute_advantages(returns, args)
            # train model
            train_model(model, observations, actions, advantages, args)

            # log statistics
            returns = [sum(eps_rew) for eps_rew in rewards]
            lengths = [len(eps_rew) for eps_rew in rewards]
            total_timesteps += sum(lengths)
            print("Iteration %d:" % (i + 1),
                  "reward mean %f±%f" % (np.mean(returns), np.std(returns)),
                  "episode length %f±%f" % (np.mean(lengths), np.std(lengths)),
                  "total timesteps", total_timesteps)
            csvwriter.writerow([time.time() - start, i,
                               np.mean(returns), np.std(returns),
                               np.max(returns), np.min(returns),
                               np.mean(lengths), np.std(lengths),
                               sum(lengths), total_timesteps])

        csvfile.close()
        model.save(os.path.join(logdir, 'model.hdf5'))

    print("Done")
