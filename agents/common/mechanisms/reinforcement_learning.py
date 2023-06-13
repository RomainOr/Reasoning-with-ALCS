"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common.RandomNumberGenerator import RandomNumberGenerator


def update_classifier_q_learning(
        cl, 
        step_reward: int, 
        max_fitness: float,
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
    max_fitness: float
        Maximum fitness from the action set
    beta_rl: float
        Learning rate of RL
    gamma: float
        Reinforcement rate
    """
    cl.r += beta_rl * (step_reward + gamma * max_fitness - cl.r)
    cl.ir += beta_rl * (step_reward - cl.ir)
