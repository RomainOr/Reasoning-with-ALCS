"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common.RandomNumberGenerator import RandomNumberGenerator

from agents.beacs.classifier_components import BEACSClassifier


def update_classifier_double_q_learning(
        cl: BEACSClassifier, 
        step_reward: int, 
        max_fitness_r: float,
        max_fitness_r_bis: float,
        beta_rl: float, 
        gamma: float
    ) -> None:
    """
    Applies adapted Double Q-Learning according to current reinforcement
    `step_reward` and back-propagated reinforcement max_fitness_r and max_fitness_r_bis.

    Parameters
    ----------
        cl: BEACSClassifier
        step_reward: int
        max_fitness_r: float
        max_fitness_r_bis: float
        beta_rl: float
        gamma: float
    """
    if RandomNumberGenerator.random() < 0.5:
        cl.err += beta_rl * (abs(step_reward + gamma * max_fitness_r_bis - cl.r) - cl.err)
        cl.r += beta_rl * (step_reward + gamma * max_fitness_r_bis - cl.r)
    else:
        cl.err += beta_rl * (abs(step_reward + gamma * max_fitness_r - cl.r_bis) - cl.err)
        cl.r_bis += beta_rl * (step_reward + gamma * max_fitness_r - cl.r_bis)
    cl.ir += beta_rl * (step_reward - cl.ir)
