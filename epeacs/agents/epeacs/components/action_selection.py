import random
from itertools import groupby

from epeacs.agents.epeacs import Classifier


def choose_action(cll, cfg, epsilon: float) -> int:
    """
    Chooses which action to use given matching set

    Parameters
    ----------
    cll: ClassifierList
        Matching set
    cfg: Configuration
        Allow to retrieve the number of possible actions
    epsilon: float
        Probability of executing exploration path

    Returns
    -------
    int 
        Action
    """
    if random.random() < epsilon:
        return explore(cll, cfg)

    return choose_fittest_action(cll, cfg)


def explore(cll, cfg, pb: float = 0.5) -> int:
    """
    Chooses action according to current exploration policy

    Parameters
    ----------
    cll: ClassifierList
        Matching set
    cfg: Configuration
        Allow to retrieve the number of possible actions
    pb: float
        probability of biased exploration

    Returns
    -------
    int 
        Action
    """
    if random.random() < pb:
        # We are in the biased exploration
        if random.random() < 0.5:
            return choose_latest_action(cll, cfg)
        else:
            return choose_action_from_knowledge_array(cll, cfg)

    return choose_random_action(cfg)


def choose_latest_action(cll, cfg) -> int:
    """
    Chooses latest executed action ("action delay bias")

    Parameters
    ----------
    cll: ClassifierList
        Matching set
    cfg: Configuration
        Allow to retrieve the number of possible actions

    Returns
    -------
    int 
        Action
    """
    last_executed_cls = None
    number_of_cls_per_action = {i: 0 for i in range(cfg.number_of_possible_actions)}
    if len(cll) > 0:
        last_executed_cls = min(cll, key=lambda cl: cl.talp)
        # If there are some actions with no classifiers - select them
        cll.sort(key=lambda cl: cl.action)
        for _action, _clss in groupby(cll, lambda cl: cl.action):
            number_of_cls_per_action[_action] = sum([cl.num for cl in _clss])
        for action, nCls in number_of_cls_per_action.items():
            if nCls == 0:
                return action
        return last_executed_cls.action
    return choose_random_action(cfg)


def choose_action_from_knowledge_array(cll, cfg) -> Classifier:
    """
    Creates 'knowledge array' that represents the average quality of the
    anticipation for each action in the current list. Chosen is
    the action, epeacs knows least about the consequences.
    Then this action is returned.

    Parameters
    ----------
    cll: ClassifierList
        Matching set
    cfg: Configuration
        Allow to retrieve the number of possible actions

    Returns
    -------
    int 
        Action
    """
    knowledge_array = {i: 0.0 for i in range(cfg.number_of_possible_actions)}

    if len(cll) > 0:
        cll.sort(key=lambda cl: cl.action)

        for _action, _clss in groupby(cll, lambda cl: cl.action):
            _classifiers = [cl for cl in _clss]
            agg_q = sum(cl.q * cl.num for cl in _classifiers)
            agg_num = sum(cl.num for cl in _classifiers)
            knowledge_array[_action] = agg_q / float(agg_num)
        by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])
        action = by_quality[0][0]
        return action

    return choose_random_action(cfg)


def choose_random_action(cfg) -> int:
    """
    Chooses one of the possible actions in the environment randomly

    Parameters
    ----------
    cfg: Configuration
        Allow to retrieve the number of possible actions

    Returns
    -------
    int
        Action
    """
    return random.randint(0, cfg.number_of_possible_actions -1)


def choose_fittest_action(cll, cfg) -> int:
    """
    Chooses the fittest action in the matching set

    Parameters
    ----------
    cll: ClassifierList
        Matching set
    cfg: Configuration
        Allow to retrieve the number of possible actions

    Returns
    -------
    int 
        Action
    """
    if len(cll) > 0:
        anticipated_change = [cl for cl in cll if cl.does_anticipate_change()]
        if len(anticipated_change) > 0:
            best_classifier = max(anticipated_change, key=lambda cl: cl.fitness)
            return best_classifier.action
    return choose_random_action(cfg)
