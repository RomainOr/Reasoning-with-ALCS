"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

# General
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# To avoid Type3 fonts in generated pdf file
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

### Provide a helper method for calculating obtained knowledge and other metrics

# Provide a wrapper for plotting

def parse_metrics_to_df(metrics_explore, metrics_trial_frequency_explore, metrics_exploit, number_of_exploit_steps):
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

def find_best_classifier(population, situation, cfg):
    unused_match_set, best_classifier, unused_max_fitness_ra, unused_max_fitness_rb = population.form_match_set(situation)
    return best_classifier

def update_matrix_index(original, tmp_x, tmp_y, action):
    if action == 0 and original[(tmp_x, tmp_y)]== 0:
        tmp_x -= 1
    elif action == 1 and original[(tmp_x, tmp_y)]== 0:
        tmp_x -= 1
        tmp_y += 1
    elif action == 2 and original[(tmp_x, tmp_y)]== 0:
        tmp_y += 1
    elif action == 3 and original[(tmp_x, tmp_y)]== 0:
        tmp_x += 1
        tmp_y += 1
    elif action == 4 and original[(tmp_x, tmp_y)]== 0:
        tmp_x += 1
    elif action == 5 and original[(tmp_x, tmp_y)]== 0:
        tmp_x += 1
        tmp_y -= 1
    elif action == 6 and original[(tmp_x, tmp_y)]== 0:
        tmp_y -= 1
    elif action == 7 and original[(tmp_x, tmp_y)]== 0:
        tmp_x -= 1
        tmp_y -= 1
    return tmp_x, tmp_y

def build_fitness_matrix(env, population, cfg):
    original = env.env.maze.matrix
    fitness = original.copy()
    # Think about more 'functional' way of doing this
    for index, x in np.ndenumerate(original):
        # Path - best classfier fitness
        if x == 0:
            perception = env.env.maze.perception(index[1], index[0])
            best_cl = find_best_classifier(population, perception, cfg)
            if best_cl:
                fitness[index] = max(best_cl.fitness, fitness[index])
                if best_cl.behavioral_sequence:
                    tmp_x, tmp_y = update_matrix_index(original, index[0], index[1], best_cl.action)
                    fitness[(tmp_x, tmp_y)] = max(fitness[(tmp_x, tmp_y)], best_cl.fitness)
                    if len(best_cl.behavioral_sequence) > 1:
                        for idx, seq in enumerate(best_cl.behavioral_sequence):
                            if idx != len(best_cl.behavioral_sequence) -1:
                                tmp_x, tmp_y = update_matrix_index(original, tmp_x, tmp_y, seq)
                                fitness[(tmp_x, tmp_y)] = max(fitness[(tmp_x, tmp_y)], best_cl.fitness)
            else:
                fitness[index] = -1
    for index, x in np.ndenumerate(original):
        # Wall - fitness = 0
        if x == 1:
            fitness[index] = 0
        # Reward - inf fitness
        if x == 9:
            fitness[index] = fitness.max() + 500
    return fitness

def build_action_matrix(env, population, cfg, fitness_matrix):
    ACTION_LOOKUP = {
        0: u'↑', 1: u'↗', 2: u'→', 3: u'↘',
        4: u'↓', 5: u'↙', 6: u'←', 7: u'↖'
    }
    original = env.env.maze.matrix
    action = original.copy().astype(str)
    # Think about more 'functional' way of doing this
    for index, x in np.ndenumerate(original):
        action[index] = ''
    for index, x in np.ndenumerate(original):
        # Path - best classfier fitness
        if x == 0:
            perception = env.env.maze.perception(index[1], index[0])
            best_cl = find_best_classifier(population, perception, cfg)
            if best_cl:
                if action[index].find(ACTION_LOOKUP[best_cl.action]) == -1:
                    action[index] += ACTION_LOOKUP[best_cl.action]
                if best_cl.behavioral_sequence:
                    for act in best_cl.behavioral_sequence:
                        action[index] += ACTION_LOOKUP[act]
            else:
                action[index] = '?'
        # Wall - fitness = 0
        if x == 1:
            action[index] = '#'
        # Reward - inf fitness
        if x == 9:
            action[index] = 'R'
    return action

def plot_policy(env, agent, cfg, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    # Handy variables
    max_x = env.env.maze.max_x
    max_y = env.env.maze.max_y
    fitness_matrix = build_fitness_matrix(env, agent.population, cfg)
    action_matrix = build_action_matrix(env, agent.population, cfg, fitness_matrix)
    # Render maze as image
    plt.imshow(fitness_matrix, interpolation='nearest', cmap='Reds', aspect='auto',
           extent=[0, max_x, max_y, 0])
    # Add labels to each cell
    for (y,x), val in np.ndenumerate(action_matrix):
        plt.text(x+0.4, y+0.5, val)
    ax.set_title("Policy", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel('x', fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel('y', fontsize=AXIS_TEXT_SIZE)
    ax.set_xlim(0, max_x)
    ax.set_ylim(max_y, 0)
    ax.set_xticks(range(0, max_x+1))
    ax.set_yticks(range(0, max_y+1))
    ax.grid(True)

def plot_knowledge(df, metrics_trial_frequency_explore, number_of_exploit_steps, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12):
    if ax is None:
        ax = plt.gca()
    explore_df = df.query("phase == 'explore'")
    explore_df.plot(y='knowledge', ax=ax, c='blue', legend=False)
    ax.set_title("Achieved knowledge", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Knowledge [%]", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylim([0, 105])

def plot_steps(df, metrics_trial_frequency_explore, number_of_exploit_steps, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12):
    if ax is None:
        ax = plt.gca()
    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")
    explore_df.plot(y='steps_in_trial', ax=ax, c='blue', linewidth=0.5, legend=False)
    exploit_df.plot(y='steps_in_trial', ax=ax, c='red', linewidth=0.5, legend=False)
    if number_of_exploit_steps:
        ax.vlines(x=len(explore_df)*metrics_trial_frequency_explore+number_of_exploit_steps[0], ymin=0, ymax=max(exploit_df['steps_in_trial'])+1, colors='black', linestyle='dashed')
        ax.vlines(x=len(explore_df)*metrics_trial_frequency_explore+number_of_exploit_steps[0]+number_of_exploit_steps[1], ymin=0, ymax=max(exploit_df['steps_in_trial'])+1, colors='black', linestyle='dashed')
    ax.set_title("Steps", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Steps", fontsize=AXIS_TEXT_SIZE)

def plot_classifiers(df, metrics_trial_frequency_explore, number_of_exploit_steps, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12, LEGEND_TEXT_SIZE=14):
    if ax is None:
        ax = plt.gca()
    explore_df = df.query("phase == 'explore'")
    explore_df.plot(y='numerosity', ax=ax, c='blue')
    explore_df.plot(y='population', ax=ax, c='green')
    explore_df.plot(y='reliable', ax=ax, c='red')
    ax.set_title("Classifiers", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Classifiers", fontsize=AXIS_TEXT_SIZE)
    ax.legend(fontsize=LEGEND_TEXT_SIZE)

def plot_performance(agent, maze, metrics_df, cfg, env_name, metrics_trial_frequency_explore, number_of_exploit_steps):
    plt.figure(figsize=(13, 10), dpi=100)
    plt.suptitle(f'BEACS Performance in {env_name} environment', fontsize=32)
    ax1 = plt.subplot(221)
    plot_policy(maze, agent, cfg, ax1)
    ax2 = plt.subplot(222)
    plot_knowledge(metrics_df,metrics_trial_frequency_explore, number_of_exploit_steps, ax2)
    ax3 = plt.subplot(223)
    plot_classifiers(metrics_df,metrics_trial_frequency_explore, number_of_exploit_steps, ax3)
    ax4 = plt.subplot(224)
    plot_steps(metrics_df,metrics_trial_frequency_explore, number_of_exploit_steps, ax4)
    plt.subplots_adjust(top=0.86, wspace=0.3, hspace=0.3)

def plot_cartpole_performance(agent, cartpole_env, metrics_df, cfg, env_name, metrics_trial_frequency_explore, number_of_exploit_steps):
    plt.figure(figsize=(13, 10), dpi=100)
    plt.suptitle(f'BEACS Performance in {env_name} environment', fontsize=32)
    ax1 = plt.subplot(221)
    plot_classifiers(metrics_df,metrics_trial_frequency_explore, number_of_exploit_steps, ax1)
    ax2 = plt.subplot(222)
    plot_steps(metrics_df,metrics_trial_frequency_explore, number_of_exploit_steps, ax2)
    plt.subplots_adjust(top=0.86, wspace=0.3, hspace=0.3)

def plot_average_cartpole_performance(trials):
    average_scores=[]
    solved = -1
    solved_averaged = 0.
    for i in range(len(trials)-100):
        check_solved = trials[i:i+100]
        average = float(sum(check_solved) / 100)
        average_scores.append(average)
        if average >= 195.0 and solved == -1:
            solved = 100+1+i
            solved_averaged = average
    if solved > 0 :
        print("Solved requirements at episode {} : average {} for {} episodes".format(
            solved, solved_averaged, 100))
    y = [i for i in range(100,100+len(average_scores))]
    plt.plot(y, average_scores)
    plt.ylabel('Average Scores')
    plt.xlabel('Episodes')
    plt.grid(True)
    plt.show()