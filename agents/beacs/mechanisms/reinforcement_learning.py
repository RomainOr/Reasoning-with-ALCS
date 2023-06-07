"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common import RandomNumberGenerator


def update_classifier_double_q_learning(
        cl, 
        step_reward: int, 
        max_fitness_ra: float,
        max_fitness_rb: float,
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
    max_fitness_ra: float
        Maximum fitness from the action set and from the Q_a function
    max_fitness_rb: float
        Maximum fitness from the action set and from the Q_b function
    beta_rl: float
        Learning rate of RL
    gamma: float
        Reinforcement rate
    """
    if RandomNumberGenerator.random() < 0.5:
        cl.err += beta_rl * (abs(step_reward + gamma * max_fitness_rb - cl.ra) - cl.err)
        cl.ra += beta_rl * (step_reward + gamma * max_fitness_rb - cl.ra)
    else:
        cl.err += beta_rl * (abs(step_reward + gamma * max_fitness_ra - cl.rb) - cl.err)
        cl.rb += beta_rl * (step_reward + gamma * max_fitness_ra - cl.rb)
    cl.ir += beta_rl * (step_reward - cl.ir)
