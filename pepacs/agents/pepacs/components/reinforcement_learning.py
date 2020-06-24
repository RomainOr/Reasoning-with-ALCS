"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

def update_classifier(
        cl, 
        step_reward: int, 
        max_fitness: float,
        beta_rl: float, 
        gamma: float
    ):
    """
    Applies Reinforcement Learning according to
    current reinforcement `reward` and back-propagated reinforcement
    `maximum_fitness`.

    Classifier parameters are updated.

    Parameters
    ----------
    cl:
        classifier with `r` and `ir` properties
    step_reward: int
        current reward obtained from the environment after executing step
    max_fitness: float
        maximum fitness - back-propagated reinforcement. Maximum fitness
        from the match set
    beta_rl: float
        learning rate of RL
    gamma: float
        reinforcement rate
    """
    _reward = step_reward + gamma * max_fitness

    # Update classifier properties
    cl.r += beta_rl * (_reward - cl.r)
    cl.ir += beta_rl * (step_reward - cl.ir)
