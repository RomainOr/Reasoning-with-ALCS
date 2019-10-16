import logging
import random
from itertools import groupby

from bacs.agents.bacs import Classifier

import numpy as np

logger = logging.getLogger(__name__)

def choose_action(cll, cfg, epsilon: float) -> Classifier:
    """
    Chooses which action to execute given classifier list (match set).

    Parameters
    ----------
    cll:
        list of classifiers
    cfg:
        Configuration that contains the number of possible actions
    epsilon: float
        Probability of executing exploration path

    Returns
    -------
    int
        number of chosen action
    """
    if random.random() < epsilon:
        logger.debug("\t\tExploration path")
        return explore(cll, cfg)

    logger.debug("\t\tExploitation path")
    #return roulette_wheel_selection(cll, cfg)
    return choose_fittest_classifier(cll, cfg)

def explore(cll, cfg, pb: float = 0.5) -> Classifier:
    """
    Chooses action according to current exploration policy

    Parameters
    ----------
    cll:
        list of classifiers
    pb: float
        probability of biased exploration

    Returns
    -------
    Classifier
        Chosen classifier
    """
    if random.random() < pb:
        # We are in the biased exploration
        if random.random() < 0.5:
            return choose_latest_action(cll, cfg)
        else:
            return choose_action_from_knowledge_array(cll, cfg)

    return choose_random_classifiers(cll, cfg)

def choose_latest_action(cll, cfg) -> Classifier:
    """
    Chooses latest executed action ("action delay bias")

    Parameters
    ----------
    cll:
        list of classifiers

    Returns
    -------
    Classifier
        Chosen classifier
    """
    last_executed_cls = None
    number_of_cls_per_action = {i: 0 for i in range(cfg.number_of_possible_actions)}
    if len(cll) > 0:
        last_executed_cls = min(cll, key=lambda cl: cl.talp)

        cll.sort(key=lambda cl: cl.action)
        for _action, _clss in groupby(cll, lambda cl: cl.action):
            number_of_cls_per_action[_action] = \
                sum([cl.num for cl in _clss])

        # If there are some actions with no classifiers - select them
        for action, nCls in number_of_cls_per_action.items():
            if nCls == 0:
                return Classifier(action=action, cfg=cfg)
        return last_executed_cls

    return choose_random_classifiers(cll, cfg)

def choose_action_from_knowledge_array(cll, cfg) -> Classifier:
    """
    Creates 'knowledge array' that represents the average quality of the
    anticipation for each action in the current list. Chosen is
    the action, BACS knows least about the consequences.

    Parameters
    ----------
    cll:
        list of classifiers
    all_actions: int
        number of all possible actions available

    Returns
    -------
    Classifier
        Chosen classifier
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
        classifiers_that_match_action = [cl for cl in cll if cl.action == action]
        if len(classifiers_that_match_action) > 0:
            return classifiers_that_match_action[0]

    return choose_random_classifiers(cll, cfg)


def choose_random_classifiers(cll, cfg) -> Classifier:
    """
    Chooses one of the possible classifiers in the matching set randomly

    Parameters
    ----------
    cll: ClassifierList
        matching set

    Returns
    -------
    Classifier
        Chosen classifier
    """
    if len(cll) > 0:
        return cll[np.random.randint(len(cll))]
    return Classifier(action=choose_random_action(cfg.number_of_possible_actions), cfg=cfg)


def roulette_wheel_selection(cll, cfg) -> Classifier:
    """
    Chooses one of the possible actions in the environment by roulette wheel selection

    Parameters
    ----------
    cll: ClassifierList
        matching set

    Returns
    -------
    Classifier
        Chosen classifier
    """
    if len(cll) > 0:
        classifier_fitness = [cl.fitness for cl in cll if cl.does_anticipate_change()]
        if len(classifier_fitness) > 0:
            total_fit = float(sum(classifier_fitness))
            classifier_relative_fitness = [f/total_fit for f in classifier_fitness]
            probabilities = [sum(classifier_relative_fitness[:i+1]) for i in range(len(classifier_relative_fitness))]
            r = np.random.random()
            for (idx, cl) in enumerate(cll):
                if r <= probabilities[idx]:
                    return cl
        return cll[np.random.randint(len(cll))]
    return choose_random_classifiers(cll, cfg)


def choose_fittest_classifier(cll, cfg) -> Classifier:
    """
    Chooses the fittest classifiers in the matching set

    Parameters
    ----------
    cll: ClassifierList
        matching set

    Returns
    -------
    Classifier
        Chosen classifier
    """
    if len(cll) > 0:
        anticipated_change = [cl for cl in cll if cl.does_anticipate_change()]
        if len(anticipated_change) > 0:
            return max(anticipated_change, key=lambda cl: cl.fitness * cl.num)
        return max(cll, key=lambda cl: cl.fitness * cl.num)
    return choose_random_classifiers(cll, cfg)


def choose_random_action(all_actions: int) -> int:
    """
    Chooses one of the possible actions in the environment randomly

    Parameters
    ----------
    all_actions: int
        number of all possible actions available

    Returns
    -------
    int
        random action number
    """
    return np.random.randint(all_actions)