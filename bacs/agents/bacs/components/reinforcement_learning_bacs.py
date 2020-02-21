def update_classifier(cl, step_reward: int, max_fitness: float,
                      beta: float, gamma: float):
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
    gamma: float
    """

    #if cl.behavioral_sequence:
    #    delta_gamma_bs = 0.01
    #    bs_ratio = len(cl.behavioral_sequence)/cl.cfg.bs_max
    #    _reward = step_reward + (gamma - delta_gamma_bs * bs_ratio ) * max_fitness
    #else :
    #    _reward = step_reward + gamma * max_fitness
        
    _reward = step_reward + gamma * max_fitness

    # Update classifier properties
    cl.r += beta * (_reward - cl.r)
    cl.ir += beta * (step_reward - cl.ir)
