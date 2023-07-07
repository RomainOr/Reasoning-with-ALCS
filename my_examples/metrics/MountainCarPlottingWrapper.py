"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

# General
from __future__ import unicode_literals

from my_examples.metrics.PlottingWrapper import plot_classifiers, plot_rewards

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# To avoid Type3 fonts in generated pdf file
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

### Provide a helper method for calculating obtained knowledge and other metrics

# Provide a wrapper for plotting

def parse_metrics_to_df(metrics_explore, metrics_trial_frequency_explore, metrics_exploit):
    # Load both metrics into data frame
    explore_df = pd.DataFrame(metrics_explore)
    exploit_df = pd.DataFrame(metrics_exploit)
    # Mark them with specific phase
    explore_df['phase'] = 'explore'
    exploit_df['phase'] = 'exploit'
    # Adjust exploit trial counter
    exploit_df['trial'] = exploit_df.apply(lambda r : r['trial']+len(explore_df)*metrics_trial_frequency_explore, axis=1)
    # Concatenate both dataframes
    df = pd.concat([explore_df, exploit_df])
    df.set_index('trial', inplace=True)
    return df

def plot_mountaincar_performance(metrics_df, env_name, metrics_trial_frequency_explore, number_of_explore_steps, number_of_exploit_steps, population, pos_bucket, vel_bucket, metrics_explore, metrics_exploit):
    plt.figure(figsize=(13, 10), dpi=100)
    plt.suptitle(f'ALCS Performance in {env_name} environment', fontsize=32)
    ax1 = plt.subplot(221)
    plot_classifiers(metrics_df, ax1)
    ax2 = plt.subplot(222)
    plot_rewards(metrics_df,metrics_trial_frequency_explore, number_of_exploit_steps, ax2)
    ax2.axhline(y = -110.0, color = 'g', linestyle = '-')
    ax3 = plt.subplot(223)
    plot_policy(ax3, population, pos_bucket, vel_bucket)
    ax4 = plt.subplot(224)
    plot_average_mountaincar_performance(ax4, metrics_explore, metrics_trial_frequency_explore, number_of_explore_steps, metrics_exploit, number_of_exploit_steps)
    ax4.axhline(y = -110.0, color = 'g', linestyle = '-')
    plt.subplots_adjust(top=0.86, wspace=0.3, hspace=0.3)

def plot_average_mountaincar_performance(ax, metrics_explore, metrics_trial_frequency_explore, NUMBER_OF_EXPLORE_TRIALS, metrics_exploit, NUMBER_OF_EXPLOIT_TRIALS_RL):
    if ax is None:
        ax = plt.gca()

    trials=[]
    avg_step_explore = 0
    for trial in metrics_explore:
        trials.append(trial['reward'])
        avg_step_explore += trial['reward']
    avg_step_explore /= NUMBER_OF_EXPLORE_TRIALS / metrics_trial_frequency_explore
    print("Average number of reward to solve the mountaincar is ",avg_step_explore,
        " for a total of ", NUMBER_OF_EXPLORE_TRIALS, " trials in EXPLORATION")

    if NUMBER_OF_EXPLOIT_TRIALS_RL:
        avg_step_exploit_rl = 0
        for trial in metrics_exploit:
            trials.append(trial['reward'])
            avg_step_exploit_rl += trial['reward']
        avg_step_exploit_rl /= NUMBER_OF_EXPLOIT_TRIALS_RL
        print("Average number of rewards to solve the mountaincar is ",avg_step_exploit_rl,
            " for a total of ", NUMBER_OF_EXPLOIT_TRIALS_RL, " trials in EXPLOITATION with Reinforcement Module")
        avg_step_exploit_rl_last_100 = 0
        for trial in metrics_exploit[-100:]:
            trials.append(trial['reward'])
            avg_step_exploit_rl_last_100 += trial['reward']
        avg_step_exploit_rl_last_100 /= 100
        print("Average number of rewards to solve the mountaincar is ",avg_step_exploit_rl_last_100,
            " for the last 100 trials in EXPLOITATION with Reinforcement Module")

    # https://github.com/openai/gym/wiki/Leaderboard#mountaincar-v0
    # MountainCar-v0 defines "solving" as getting average reward of -110.0 over 100 consecutive trials.
    average_scores=[]
    solved = -1
    solved_averaged = 0.
    for i in range(len(trials)-99):
        check_solved = trials[i:i+100]
        average = float(sum(check_solved) / 100)
        average_scores.append(average)
        if average >= -110.0 and solved == -1:
            solved = 100+1+i
            solved_averaged = average
    if solved > 0 :
        print("Solved requirements at episode {}: average {} for {} episodes".format(
            solved, solved_averaged, 100))
    y = [i for i in range(100,100+len(average_scores))]
    if len(average_scores) > 0:
        print("Maximum average achieved: {} for 100 episodes".format(max(average_scores)))
        ax.plot(y, average_scores)
    ax.set_title("Moving average", fontsize=18)
    ax.set_ylabel('Average Scores', fontsize=12)
    ax.set_xlabel('Episodes', fontsize=12)


def plot_policy(ax, population, pos_bucket, vel_bucket, pos_range = [-1.2, 0.6], vel_range = [-0.07, 0.07]):
    ACTION_LOOKUP = {
        0: u'←', 1: u'Ø', 2: u'→'
    }
    fitness_matrix = np.zeros([vel_bucket, pos_bucket])
    for (y, x), val in np.ndenumerate(fitness_matrix):
        best_cl = population.find_best_classifier((str(x+1), str(y+1)), have_to_anticipate_changes=False)
        if best_cl:
            fitness_matrix[y][x] = best_cl.fitness
        else:
            fitness_matrix[y][x] = 0
    policy_matrix = np.empty([pos_bucket, vel_bucket]).astype(str)
    for (x, y), val in np.ndenumerate(policy_matrix):
        policy_matrix[x][y] = ''
        best_cl = population.find_best_classifier((str(x+1), str(y+1)), have_to_anticipate_changes=False)
        if best_cl:
            policy_matrix[x][y] = ACTION_LOOKUP[best_cl.action]
        else:
            policy_matrix[x][y] = '?'
    x_buckets, x_step = np.linspace(pos_range[0], pos_range[1], num=pos_bucket, endpoint=False, retstep=True)
    y_buckets, y_step =np.linspace(vel_range[0], vel_range[1], num=vel_bucket, endpoint=False, retstep=True)
    ax.imshow(fitness_matrix, cmap='Reds', origin='lower', extent=[pos_range[0],pos_range[1],vel_range[0],vel_range[1]], aspect=x_step/y_step)
    for (x, y), val in np.ndenumerate(policy_matrix):
        ax.text(pos_range[0] + x * x_step + x_step/2.25, vel_range[0] + y * y_step + y_step/2.25, val)
    ax.set_title("Policy", fontsize=18)
    ax.set_xlabel('Position', fontsize=12)
    ax.set_ylabel('Velocity', fontsize=12)
    ax.set_xlim(pos_range[0], pos_range[1])
    ax.set_ylim(vel_range[0], vel_range[1])
    x_buckets_label = ["{:.2f}".format(x) for x in x_buckets] + [str(pos_range[1])]
    ax.set_xticks(ticks=list(x_buckets) + [pos_range[1]], labels=x_buckets_label, rotation=45)
    ax.set_yticks(list(y_buckets) + [vel_range[1]])
    ax.grid(True)