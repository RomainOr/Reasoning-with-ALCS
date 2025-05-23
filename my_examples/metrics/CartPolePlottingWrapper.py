"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

# General
from __future__ import unicode_literals

from my_examples.metrics.CartPoleMetrics import _check_cartpole_solved_requirement
from my_examples.metrics.PlottingWrapper import plot_classifiers, plot_steps

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

def plot_cartpole_performance(agent, cartpole_env, metrics_df, cfg, env_name, metrics_trial_frequency_explore, number_of_exploit_steps):
    plt.figure(figsize=(13, 10), dpi=100)
    plt.suptitle(f'ALCS Performance in {env_name} environment', fontsize=32)
    ax1 = plt.subplot(221)
    plot_classifiers(metrics_df, ax1)
    ax2 = plt.subplot(222)
    plot_steps(metrics_df,metrics_trial_frequency_explore, number_of_exploit_steps, ax2)
    plt.subplots_adjust(top=0.86, wspace=0.3, hspace=0.3)

def plot_average_cartpole_performance(average_scores):
    y = [i for i in range(100,100+len(average_scores))]
    plt.plot(y, average_scores)
    plt.ylabel('Average Scores')
    plt.xlabel('Episodes')
    plt.grid(True)
    plt.show()