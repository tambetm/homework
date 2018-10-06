import argparse
import os
import csv
import json
import time
import gym
import numpy as np


def create_model(observation_space, action_space, args):
    assert isinstance(observation_space, gym.spaces.Box)
    assert isinstance(action_space, gym.spaces.Box) \
        or isinstance(action_space, gym.spaces.Discrete)

    # TODO: Use given observation space, action space and command line parameters to create a model.
    # NB! Depending if action space is Discerete or Box you need different outputs and loss function.
    #     For Discrete you need to output probabilities of actions and use cross-entropy loss.
    #     For Box you need to output means of Gaussians and use mean squared error loss.

    # YOUR CODE HERE
    raise NotImplementedError

    return model


def train_model(model, observations, actions, advantages, args):
    # TODO: Use given observations, actions and advantages to train the model.

    # YOUR CODE HERE
    raise NotImplementedError


def sample_trajectories(env, model, args):
    max_steps = args.max_timesteps or env.spec.timestep_limit

    observations = []
    actions = []
    rewards = []
    total_steps = 0
    while total_steps < args.batch_size:
        episode_observations = []
        episode_actions = []
        episode_rewards = []

        obs = env.reset()
        done = False
        steps = 0
        while not done:
            # TODO: Use your model to predict action for given observation.
            # NB! You need to sample the action from probability distribution!

            # YOUR CODE HERE
            raise NotImplementedError

            episode_observations.append(obs)
            episode_actions.append(action)
            obs, reward, done, _ = env.step(action)
            episode_rewards.append(reward)
            steps += 1
            if args.render:
                env.render()
            if steps >= max_steps:
                break

        observations.append(episode_observations)
        actions.append(episode_actions)
        rewards.append(episode_rewards)
        total_steps += steps

    return observations, actions, rewards


def compute_advantages(rewards, args):
    # TODO: Compute advantages from given rewards.
    # NB! Use args.discount for discounting future rewards.
    # NB! Depending on args.reward_to_go calculate either total episode reward or future reward for each timestep.
    # NB! Depending on args.dont_normalize_advantages normalize advantages to 0 mean and 1 standard deviation.

    # YOUR CODE HERE
    raise NotImplementedError

    assert len(advantages) == sum([len(rew) for rew in rewards])
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
    #parser.add_argument('--nn_baseline', '-bl', action='store_true')
    #parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--hidden_layers', '-l', type=int, default=2)
    parser.add_argument('--hidden_nodes', '-s', type=int, default=64)
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
            # compute advantages
            advantages = compute_advantages(rewards, args)
            # flatten observations and actions
            observations = np.concatenate(observations)
            actions = np.concatenate(actions)
            # train model
            train_model(model, observations, actions, advantages, args)

            # log statistics
            returns = [sum(eps_rew) for eps_rew in rewards]
            lengths = [len(eps_rew) for eps_rew in rewards]
            total_timesteps += len(observations)
            print("Iteration %d:" % (i + 1),
                  "reward mean %f±%f" % (np.mean(returns), np.std(returns)),
                  "episode length %f±%f" % (np.mean(lengths), np.std(lengths)),
                  "total timesteps", total_timesteps)
            csvwriter.writerow([time.time() - start, i,
                               np.mean(returns), np.std(returns),
                               np.max(returns), np.min(returns),
                               np.mean(lengths), np.std(lengths),
                               len(observations), total_timesteps])

        csvfile.close()
        # TODO: Optional - save the model for later testing.

    print("Done")
