import pickle
import numpy as np
import argparse
import os
from DDPG import DDPG
from SCGEnv import SCGEnv
from copy import deepcopy
from SCGLabelingRL.utils import get_output_folder
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from utils import convert_from_sample_to_ms, cluster_signals
import math
import pandas as pd
import random

import warnings

# Suppress specific warning
warnings.simplefilter("ignore", UserWarning)

def train(num_episodes, agent, env, output_dir, is_HF_available=False):
    agent.is_training = True
    episode = 0
    all_rewards = []
    actual_episode_index = None
    all_rmse_ms_errors = []
    train_results = {'episode': [],
                     'boundary_reward': [],
                     'extremum_reward': [],
                     'consistency_reward': [],
                     'dtw_reward': [],
                     'correct_label': [],
                     'action': [],
                     'error_ms': [],
                     }
    if is_HF_available:   # add the human feedback to your dtw database
        # take one sample from each episode as a starting point
        num_HF_episodes = 30
        for hf_episode in range(num_HF_episodes):
            hf_index = random.randint(0, env.beat_length - 1)
            signal = env.episode_dict[hf_episode][hf_index]
            label = env.episode_label_dict[hf_episode][env.label_index][hf_index]
            env.add_to_dtw_database(signal, label, env.scg_label_type)
            # todo: remove redundant samples from dtw data base

        env.visualize_dtw_database()
        # remove redundant samples
        env.dtw_database = cluster_signals(env.dtw_database, n_clusters=5)
        env.visualize_dtw_database()

    while episode < num_episodes:   # episodes basically
        print("Episode: ", episode)
        observation = None
        done = False
        episode_reward = 0
        num_episode_steps = 0
        episode_predictions = []
        episode_labels = []
        while not done:
            if len(env.dtw_database) > 50:
                env.dtw_database = cluster_signals(env.dtw_database, n_clusters=25)
                env.visualize_dtw_database()


            # reset if it is the start of the episode
            num_episode_steps += 1
            if observation is None:
                observation, actual_episode_index = env.reset()
                agent.reset(observation)


            # agent picks action
            if episode <= args.warmup:
                action = agent.random_action()   # not sure if completely random is a good idea
            else:
                action = agent.select_action(observation)

            # env response with next observation, reward, terminate, info
            if episode <= args.warmup:
                observation, rewards, done, info = env.step(action, is_HF_available=False, episode=episode)
            else:
                observation, rewards, done, info = env.step(action, is_HF_available=is_HF_available, episode=episode)
                # save what you want to save into results
                if info['correct_label'] is not None:
                    train_results['episode'].append(actual_episode_index)
                    train_results['boundary_reward'].append(info['boundary_orig'])
                    train_results['extremum_reward'].append(info['extremum_orig'])
                    train_results['consistency_reward'].append(info['consistency_orig'])
                    train_results['dtw_reward'].append(info['dtw_orig'])
                    train_results['correct_label'].append(info['correct_label'])
                    train_results['action'].append(action)
                    beat_error = convert_from_sample_to_ms(info['correct_label'] - action,
                                                           env.sampling_rate, env.downsampling_fac)
                    train_results['error_ms'].append(beat_error)





            if info['correct_label'] is not None:
                episode_predictions.append(action)
                episode_labels.append(info['correct_label'])

            reward = rewards['total']
            # agent observes and updates policy
            agent.observe(reward, observation, done)

            if episode > args.warmup + 100:
                if num_episode_steps == 300:
                    # plot first 300 dimension of observation which is the beat
                    plt.plot(observation[:300])
                    # now title will be the action and I want a vertical line at the action
                    plt.title(f'Action: {action}, Cons-R: {rewards["consistency"]}, Ext-R: {rewards["extremum"]}, Bound-R: {rewards["boundary"]}, DTW-R: {rewards["dtw"]}')
                    plt.axvline(x=action, color='r', linestyle='--')
                    plt.axvline(x=info['correct_label'], color='g', linestyle='--')

                    plt.show()

            if episode > args.warmup:
                agent.update_policy()


            episode_reward += reward
            if done:  # at the end of the episode
                agent.memory.append(
                    observation,
                    agent.select_action(observation),
                    0.,
                    False
                )
        if episode > args.warmup:
            print("aha sorun: ", episode_reward)
            all_rewards.append(episode_reward / num_episode_steps)
            print("Episode reward: ", episode_reward / num_episode_steps)
            rmse_error = mean_squared_error(episode_predictions, episode_labels, squared=False)
            rmse_error_in_ms = convert_from_sample_to_ms(rmse_error, env.sampling_rate, env.downsampling_fac)
            all_rmse_ms_errors.append(rmse_error_in_ms)
            print("Episode RMSE: ", rmse_error_in_ms, "ms")
        if episode % 5 == 0:
            agent.save_model(output_dir)
            # save results
            df = pd.DataFrame(train_results)
            # Define the CSV file path
            file_path = os.path.join(output_dir, f"training_out.csv")
            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)

            # generate plots for all_rewards and all_rmse_ms_errors and save them to the output_dir
            plt.plot(all_rewards)
            plt.title("Reward Evolution Plot")
            # xlabel should be episodes
            plt.xlabel("Episodes")
            plt.ylabel("Rewards")
            plt.savefig(os.path.join(output_dir, f"rewards.png"))
            plt.close()

            plt.plot(all_rmse_ms_errors)
            plt.title("RMSE Error Evolution Plot")
            # xlabel should be episodes
            plt.xlabel("Episodes")
            plt.ylabel("RMSE Error (ms)")
            plt.savefig(os.path.join(output_dir, f"rmse_errors.png"))

        episode +=1





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SCG RL method")
    parser.add_argument('--machine', default='local', type=str, help='server or local')
    parser.add_argument('--env', default='SCGEnv', type=str, help='SCGEnv environment')
    parser.add_argument('--scg_label_type', default='AO', type=str, help='AO or AC signal type')
    parser.add_argument('--mode', default='train', type=str, help='train or test mode selector')
    parser.add_argument('--hidden1', default=256, type=int, help='hidden num of first fully connected')
    parser.add_argument('--hidden2', default=128, type=int, help='hidden num of second fully connected')
    parser.add_argument('--rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--l_rate_actor', default=0.001, type=float, help='learning rate for the actor')
    parser.add_argument('--l_rate_critic', default=0.001, type=float, help='learning rate for the critic')
    parser.add_argument('--prate', default=0.0001, type=float, help='policy net learning rate (only for DDPG)')
    parser.add_argument('--warmup', default=100, type=int, help='time without training but only filling the replay memory')
    parser.add_argument('--bsize', default=64, type=int, help='batch size')
    parser.add_argument('--memsize', default=10000, type=int, help='memory size')
    parser.add_argument('--tau', default=0.001, type=float, help='moving average for target network')
    parser.add_argument('--ou_theta', default=0.15, type=float, help='noise theta')
    parser.add_argument('--ou_sigma', default=0.2, type=float, help='noise sigma')
    parser.add_argument('--ou_mu', default=0.0, type=float, help='noise mu')
    parser.add_argument('--max_episode_length', default=300, type=int, help='')
    parser.add_argument('--num_episodes', default=500, type=int,
                        help='num episodes to train')
    parser.add_argument('--train_iter', default=200000, type=int, help='train iters each timestep')
    parser.add_argument('--epsilon', default=50000, type=int, help='linear decay of exploration policy')
    parser.add_argument('--seed', default=-1, type=int, help='')
    parser.add_argument('--action_noise_mean', default=20.0, type=float, help='mean noise to be added to selected action (in ms)')
    parser.add_argument('--action_noise_std', default=20.0, type=float, help='std noise to be added to selected action (in ms)')
    parser.add_argument('--num_past_detections', default=5, type=int, help='number of previous detections used for consistency')
    parser.add_argument('--extremum_type', default='peak', type=str, help='extremum to be tracked as ao/ac')
    parser.add_argument('--use_prominence', default=False, type=bool, help='whether to use prominence or not in peakness reward')
    parser.add_argument('--window_length', default=1, type=int, help='window length for the memory')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size for training')
    parser.add_argument('--discount', default=0.99, type=float, help='')
    parser.add_argument('--output_dir', default='Output', type=str, help='output directory name')
    parser.add_argument('--is_HF_available', default=True, type=bool, help='whether to use Human Feedback or not. Requires labeled dataset for now and will require external labeling later')
    parser.add_argument('--sampling_rate', default=2000, type=int, help='sampling rate of the SCG signal')
    parser.add_argument('--downsampling_factor', default=8, type=int, help='downsampling factor for the SCG signal')
    args = parser.parse_args()


    # convert args.action_noise_mean and args.action_noise_std to ms
    args.action_noise_mean = convert_from_sample_to_ms(args.action_noise_mean, args.sampling_rate, args.downsampling_factor)
    args.action_noise_std = convert_from_sample_to_ms(args.action_noise_std, args.sampling_rate, args.downsampling_factor)
    # todo: pretrain the CNN so that you can speed up RL algorithm analysis

    # todo: pick nearest peak as decision before rmse calculation
    # todo: pick more diverse points for your dtw database
    # todo: dtw window length can be as long as the boundary picked
    # todo: think about the confidence measure
    #todo: need to add standardization
    if args.machine == 'server':
        project_dir = r"/home/cmyldz/GaTech Dropbox/Cem Yaldiz/RLSCGLabeling"
    else:
        project_dir = r"C:\Users\Cem Okan\GaTech Dropbox\Cem Yaldiz\RLSCGLabeling"
    output_dir = os.path.join(project_dir, args.output_dir)
    output_dir = get_output_folder(output_dir, args.env)

    scg_env = SCGEnv(project_dir, scg_label_type=args.scg_label_type, sampling_rate=args.sampling_rate,
                     downsampling_fac=args.downsampling_factor, extremum_type=args.extremum_type,
                     num_past_detections=args.num_past_detections, use_prominence=args.use_prominence,
                     is_HF_available=args.is_HF_available)

    # fix seed
    seed = 23
    np.random.seed(seed)

    n_past_detections = 5
    env_beat_length = scg_env.beat_length
    n_bounds = 2  # bounds of ao ac search region

    n_actions = 1  # only one action - AO or AC label
    n_states = env_beat_length + n_bounds + n_past_detections  # our state will consist of the whole signal + boundaries + past detections
    # why? - We want to add boundary to make sure that the model will return prediction from that region
    # We want to add past detections for consistency


    agent = DDPG(n_states, n_actions, args=args, action_upper_range=env_beat_length)
    if args.mode == 'train':
        train(num_episodes=args.num_episodes, agent=agent, env=scg_env, output_dir=output_dir, is_HF_available=args.is_HF_available)
    # initialize your actor
