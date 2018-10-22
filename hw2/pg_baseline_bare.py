import argparse
import os
import csv
import json
import time
import gym
import numpy as np


def create_policy_model(observation_space, action_space, args):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box) \
        or isinstance(action_space, gym.spaces.Discrete)

    # TODO: Use given observation space, action space and command line parameters to create policy model.
    # NB! Use args.hidden_layers and args.hidden_nodes for the number of hidden layers and nodes.
    # NB! Depending if action space is Discerete or Box you need different outputs and loss function.
    #     For Discrete you need to output probabilities of actions and use cross-entropy loss.
    #     For Box you need to output means of Gaussians and use mean squared error loss.

    # YOUR CODE HERE
    raise NotImplementedError

    return model


def create_baseline_model(observation_space, args):
    # TODO: Use given observation space and command line parameters to create baseline model.
    # NB! Use mean squared error as baseline loss.

    # YOUR CODE HERE
    raise NotImplementedError

    return model


def train_policy_model(model, observations, actions, advantages, args):
    # TODO: Use given observations, actions and advantages to train the policy model.
    # NB! Observations are the inputs, actions are the targets, advantages are the weights for loss function.

    # NB! Use actions as targets to policy and returns as targets to baseline branch.
    # NB! Use advantages to weight only policy loss, no need to weight baseline loss.

    # YOUR CODE HERE
    raise NotImplementedError


def train_baseline_model(model, observations, returns, args):
    # TODO: Use given observations and returns to train the baseline model.
    # NB! Observations are the inputs, returns are the targets.
    # NB! Normalize returns to zero mean and standard deviation 1 before using them as targets!

    # YOUR CODE HERE
    raise NotImplementedError

    return loss


def predict_baselines(model, observations):
    # TODO: Use given observations to predict baseline returns.

    # YOUR CODE HERE
    raise NotImplementedError

    return baselines


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
            # TODO: Use your model to predict action for given observation.
            # NB! You need to sample the action from probability distribution!

            # YOUR CODE HERE
            raise NotImplementedError

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
    # TODO: Compute returns for each timestep.
    # NB! Use args.discount for discounting future rewards.
    # NB! Depending on args.reward_to_go calculate either total episode reward or future reward for each timestep.

    # YOUR CODE HERE
    raise NotImplementedError

    return returns


def compute_advantages(returns, baselines, args):
    # TODO: Compute advantages as difference between returns and baselines.
    # NB! Depending on args.dont_normalize_advantages normalize advantages to 0 mean and 1 standard deviation.
    # NB! Rescale baselines to have the same mean and standard deviation as returns before calculating advantages!

    # YOUR CODE HERE
    raise NotImplementedError

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
    parser.add_argument('--baseline_batch_size', type=int, default=32)
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
        model_policy = create_policy_model(env.observation_space, env.action_space, args)
        model_baseline = create_baseline_model(env.observation_space, args)

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
            observations, actions, rewards = sample_trajectories(env, model_policy, args)
            # compute returns
            returns = compute_returns(rewards, args)
            # flatten observations, actions and returns
            observations = np.concatenate(observations)
            actions = np.concatenate(actions)
            returns = np.concatenate(returns)
            # predict baselines
            baselines = predict_baselines(model_baseline, observations)
            # compute advantages
            advantages = compute_advantages(returns, baselines, args)
            # train baseline model
            loss = train_baseline_model(model_baseline, observations, returns, args)
            # train policy model
            train_policy_model(model_policy, observations, actions, advantages, args)

            # log statistics
            returns = [sum(eps_rew) for eps_rew in rewards]
            lengths = [len(eps_rew) for eps_rew in rewards]
            total_timesteps += sum(lengths)
            print("Iteration %d:" % (i + 1),
                  "reward mean %f±%f" % (np.mean(returns), np.std(returns)),
                  "episode length %f±%f" % (np.mean(lengths), np.std(lengths)),
                  "total timesteps", total_timesteps,
                  "baseline loss", loss)
            csvwriter.writerow([time.time() - start, i,
                               np.mean(returns), np.std(returns),
                               np.max(returns), np.min(returns),
                               np.mean(lengths), np.std(lengths),
                               sum(lengths), total_timesteps])

        csvfile.close()
        # TODO: Optional - save the model for later testing.

    print("Done")
