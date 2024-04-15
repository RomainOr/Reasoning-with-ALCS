"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


def update_classifier_q_learning(
        cl, 
        step_reward: int, 
        max_fitness: float,
        beta_rl: float, 
        gamma: float
    ) -> None:
    """
    Applies Q-Learning according to current reinforcement
    `step_reward` and back-propagated reinforcement `max_fitness`.

    Parameters
    ----------
        cl
        step_reward: int
        max_fitness: float
        beta_rl: float
        gamma: float
    """
    cl.r += beta_rl * (step_reward + gamma * max_fitness - cl.r)
    cl.ir += beta_rl * (step_reward - cl.ir)
