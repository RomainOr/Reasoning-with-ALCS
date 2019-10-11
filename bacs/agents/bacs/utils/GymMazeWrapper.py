# General
from __future__ import unicode_literals

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

# To avoid Type3 fonts in generated pdf file
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

from bacs.metrics import population_metrics

### Provide a helper method for calculating obtained knowledge

# TODO update with behavioral sequences

def _maze_metrics(pop, env):

    def _maze_knowledge(population, environment) -> float:
        transitions = environment.env.get_all_possible_transitions()
        # Take into consideration only reliable classifiers
        reliable_classifiers = [c for c in population if c.is_reliable()]
        # Count how many transitions are anticipated correctly
        nr_correct = 0
        # For all possible destinations from each path cell
        for start, action, end in transitions:
            p0 = environment.env.maze.perception(*start)
            p1 = environment.env.maze.perception(*end)
            if any([True for cl in reliable_classifiers
                    if cl.predicts_successfully(p0, action, p1)]):
                nr_correct += 1
        return nr_correct / len(transitions) * 100.0

    metrics = {
        'knowledge': _maze_knowledge(pop, env)
    }

    # Add basic population metrics
    metrics.update(population_metrics(pop, env))

    return metrics

# Provide a wrapper for plotting

def parse_metrics_to_df(explore_metrics, exploit_metrics):
    def extract_details(row):
        row['trial'] = row['trial']
        row['steps'] = row['steps_in_trial']
        row['numerosity'] = row['numerosity']
        row['reliable'] = row['reliable']
        row['knowledge'] = row['knowledge']
        return row
    # Load both metrics into data frame
    explore_df = pd.DataFrame(explore_metrics)
    exploit_df = pd.DataFrame(exploit_metrics)
    # Mark them with specific phase
    explore_df['phase'] = 'explore'
    exploit_df['phase'] = 'exploit'
    # Extract details
    explore_df = explore_df.apply(extract_details, axis=1)
    exploit_df = exploit_df.apply(extract_details, axis=1)
    # Adjuts exploit trial counter
    exploit_df['trial'] = exploit_df.apply(lambda r: r['trial']+len(explore_df), axis=1)
    # Concatenate both dataframes
    df = pd.concat([explore_df, exploit_df])
    df.set_index('trial', inplace=True)
    return df

def find_best_classifier(population, situation, cfg):
    match_set = population.form_match_set(situation)
    anticipated_change_cls = [cl for cl in match_set if cl.does_anticipate_change()]
    if (len(anticipated_change_cls) > 0):
        return max(anticipated_change_cls, key=lambda cl: cl.fitness * cl.num)
    return None

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
                fitness[index] = best_cl.fitness
            else:
                fitness[index] = -1
        # Wall - fitness = 0
        if x == 1:
            fitness[index] = 0
        # Reward - inf fitness
        if x == 9:
            fitness[index] = fitness.max () + 500
    return fitness

def build_action_matrix(env, population, cfg):
    ACTION_LOOKUP = {
        0: u'↑', 1: u'↗', 2: u'→', 3: u'↘',
        4: u'↓', 5: u'↙', 6: u'←', 7: u'↖'
    }
    original = env.env.maze.matrix
    action = original.copy().astype(str)
    # Think about more 'functional' way of doing this
    for index, x in np.ndenumerate(original):
        # Path - best classfier fitness
        if x == 0:
            perception = env.env.maze.perception(index[1], index[0])
            best_cl = find_best_classifier(population, perception, cfg)
            if best_cl:
                action[index] = ACTION_LOOKUP[best_cl.action]
            else:
                action[index] = '?'
        # Wall - fitness = 0
        if x == 1:
            action[index] = '\#'
        # Reward - inf fitness
        if x == 9:
            action[index] = 'R'
    return action

def plot_policy(env, agent, cfg, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12):
    if ax is None:
        ax = plt.gca()
    ax.set_aspect("equal")
    # Handy variables
    maze_countours = env.env.maze.matrix
    max_x = env.env.maze.max_x
    max_y = env.env.maze.max_y
    fitness_matrix = build_fitness_matrix(env, agent.population, cfg)
    action_matrix = build_action_matrix(env, agent.population, cfg)
    # Render maze as image
    plt.imshow(fitness_matrix, interpolation='nearest', cmap='Reds', aspect='auto',
           extent=[0, max_x, max_y, 0])
    # Add labels to each cell
    for (y,x), val in np.ndenumerate(action_matrix):
        plt.text(x+0.4, y+0.5, "${}$".format(val))
    ax.set_title("Policy", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel('x', fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel('y', fontsize=AXIS_TEXT_SIZE)
    ax.set_xlim(0, max_x)
    ax.set_ylim(1+max_y, 0)
    ax.set_xticks(range(0, max_x+1))
    ax.set_yticks(range(0, max_y+1))
    ax.grid(True)

def plot_knowledge(df, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12):
    if ax is None:
        ax = plt.gca()
    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")
    explore_df['knowledge'].plot(ax=ax, c='blue')
    exploit_df['knowledge'].plot(ax=ax, c='red')
    ax.axvline(x=len(explore_df), c='black', linestyle='dashed')
    ax.set_title("Achieved knowledge", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Knowledge [%]", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylim([0, 105])

def plot_steps(df, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12):
    if ax is None:
        ax = plt.gca()
    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")
    explore_df['steps'].plot(ax=ax, c='blue', linewidth=.5)
    exploit_df['steps'].plot(ax=ax, c='red', linewidth=0.5)
    ax.axvline(x=len(explore_df), c='black', linestyle='dashed')
    ax.set_title("Steps", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Steps", fontsize=AXIS_TEXT_SIZE)

def plot_classifiers(df, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12, LEGEND_TEXT_SIZE=14):
    if ax is None:
        ax = plt.gca()
    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")
    df['numerosity'].plot(ax=ax, c='blue')
    df['reliable'].plot(ax=ax, c='red')
    ax.axvline(x=len(explore_df), c='black', linestyle='dashed')
    ax.set_title("Classifiers", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Classifiers", fontsize=AXIS_TEXT_SIZE)
    ax.legend(fontsize=LEGEND_TEXT_SIZE)

def plot_performance(agent, maze, metrics_df, cfg, env_name):
    plt.figure(figsize=(13, 10), dpi=100)
    plt.suptitle(f'BACS Performance in {env_name} environment', fontsize=32)
    ax1 = plt.subplot(221)
    plot_policy(maze, agent, cfg, ax1)
    ax2 = plt.subplot(222)
    plot_knowledge(metrics_df, ax2)
    ax3 = plt.subplot(223)
    plot_classifiers(metrics_df, ax3)
    ax4 = plt.subplot(224)
    plot_steps(metrics_df, ax4)
    plt.subplots_adjust(top=0.86, wspace=0.3, hspace=0.3)