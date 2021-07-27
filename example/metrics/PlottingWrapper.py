"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

# General
from __future__ import unicode_literals

import matplotlib
import matplotlib.pyplot as plt

# To avoid Type3 fonts in generated pdf file
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def plot_steps(df, metrics_trial_frequency_explore, number_of_exploit_steps, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12):
    if ax is None:
        ax = plt.gca()
    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")
    explore_df.plot(y='steps_in_trial', ax=ax, c='blue', linewidth=0.5, legend=False)
    exploit_df.plot(y='steps_in_trial', ax=ax, c='red', linewidth=0.5, legend=False)
    if number_of_exploit_steps and len(number_of_exploit_steps) > 0:
        ax.vlines(x=len(explore_df)*metrics_trial_frequency_explore, ymin=min(min(explore_df['steps_in_trial']), min(exploit_df['steps_in_trial']))-1, ymax=max(max(explore_df['steps_in_trial']), max(exploit_df['steps_in_trial']))+1, colors='black', linestyle='dashed')
        ax.vlines(x=len(explore_df)*metrics_trial_frequency_explore+number_of_exploit_steps[0], ymin=min(exploit_df['steps_in_trial'])-1, ymax=max(exploit_df['steps_in_trial'])+1, colors='black', linestyle='dashed')
        if len(number_of_exploit_steps) > 1:
            ax.vlines(x=len(explore_df)*metrics_trial_frequency_explore+number_of_exploit_steps[0]+number_of_exploit_steps[1], ymin=min(exploit_df['steps_in_trial'])-1, ymax=max(exploit_df['steps_in_trial'])+1, colors='black', linestyle='dashed')
    ax.set_title("Steps", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Steps", fontsize=AXIS_TEXT_SIZE)

def plot_rewards(df, metrics_trial_frequency_explore, number_of_exploit_steps, ax=None, TITLE_TEXT_SIZE=18, AXIS_TEXT_SIZE=12):
    if ax is None:
        ax = plt.gca()
    explore_df = df.query("phase == 'explore'")
    exploit_df = df.query("phase == 'exploit'")
    explore_df.plot(y='reward', ax=ax, c='blue', linewidth=0.5, legend=False)
    exploit_df.plot(y='reward', ax=ax, c='red', linewidth=0.5, legend=False)
    if number_of_exploit_steps and len(number_of_exploit_steps) > 0:
        ax.vlines(x=len(explore_df)*metrics_trial_frequency_explore, ymin=min(min(explore_df['reward']), min(exploit_df['reward']))-1, ymax=max(max(explore_df['reward']), max(exploit_df['reward']))+1, colors='black', linestyle='dashed')
        ax.vlines(x=len(explore_df)*metrics_trial_frequency_explore+number_of_exploit_steps[0], ymin=min(explore_df['reward'])-1, ymax=max(exploit_df['reward'])+1, colors='black', linestyle='dashed')
        if len(number_of_exploit_steps) > 1:
            ax.vlines(x=len(explore_df)*metrics_trial_frequency_explore+number_of_exploit_steps[0]+number_of_exploit_steps[1], ymin=min(exploit_df['reward'])-1, ymax=max(exploit_df['steps_in_trial'])+1, colors='black', linestyle='dashed')
    ax.set_title("Rewards", fontsize=TITLE_TEXT_SIZE)
    ax.set_xlabel("Trial", fontsize=AXIS_TEXT_SIZE)
    ax.set_ylabel("Rewards", fontsize=AXIS_TEXT_SIZE)

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