"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common.RandomNumberGenerator import RandomNumberGenerator


def update_classifier_double_q_learning(
        cl, 
        step_reward: int, 
        max_fitness_r: float,
        max_fitness_r_bis: float,
        beta_rl: float, 
        gamma: float
    ) -> None:
    """
    Applies adapted Double Q-Learning according to current reinforcement
    `reward` and back-propagated reinforcement `maximum_fitness`.

    Parameters
    ----------
    cl:
        Classifier with `r` and `ir` properties
    step_reward: int
        Current reward obtained from the environment after executing step
    max_fitness_r: float
        Maximum fitness from the action set and from the Q_r function
    max_fitness_r_bis: float
        Maximum fitness from the action set and from the Q_r_bis function
    beta_rl: float
        Learning rate of RL
    gamma: float
        Reinforcement rate
    """
    if RandomNumberGenerator.random() < 0.5:
        cl.err += beta_rl * (abs(step_reward + gamma * max_fitness_r_bis - cl.r) - cl.err)
        cl.r += beta_rl * (step_reward + gamma * max_fitness_r_bis - cl.r)
    else:
        cl.err += beta_rl * (abs(step_reward + gamma * max_fitness_r - cl.r_bis) - cl.err)
        cl.r_bis += beta_rl * (step_reward + gamma * max_fitness_r - cl.r_bis)
    cl.ir += beta_rl * (step_reward - cl.ir)
