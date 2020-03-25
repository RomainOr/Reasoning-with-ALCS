"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

def update_classifier(
        cl, 
        step_reward: int, 
        max_fitness: float,
        beta: float, 
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
    beta: float
        learning rate
    gamma: float
        reinforcement rate
    """

    if cl.behavioral_sequence:
        delta_gamma_bs = 0.001
        bs_ratio = len(cl.behavioral_sequence)/cl.cfg.bs_max
        _reward = step_reward + (gamma - delta_gamma_bs * bs_ratio ) * max_fitness
    else :
        _reward = step_reward + gamma * max_fitness

    # Update classifier properties
    cl.r += beta * (_reward - cl.r)
    cl.ir += beta * (step_reward - cl.ir)
