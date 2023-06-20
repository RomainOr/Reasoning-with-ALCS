"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from itertools import groupby

from agents.common.BaseClassifiersList import BaseClassifiersList
from agents.common.BaseConfiguration import BaseConfiguration
from agents.common.RandomNumberGenerator import RandomNumberGenerator
from agents.common.classifier_components.BaseClassifier import BaseClassifier


def choose_classifier(
        cll: BaseClassifiersList,
        cfg: BaseConfiguration
    ) -> BaseClassifier:
    """
    Chooses which classifier to use given matching set through an
    epsilon greedy method.

    Parameters
    ----------
        cll: BaseClassifiersList
        cfg: BaseConfiguration

    Returns
    -------
    BaseClassifier
    """
    if RandomNumberGenerator.random() < cfg.epsilon:
        return explore(cll, cfg)
    return choose_fittest_classifier(cll, cfg)


def explore(
        cll: BaseClassifiersList,
        cfg: BaseConfiguration,
        pb: float = 0.5
    ) -> BaseClassifier:
    """
    Chooses classifier according to current exploration policy.
    There is pb probability to choose a random classifier.
    Otherwise, in double biased exploration, there is 
    (1-pb)/2 to use one of the bias.

    Parameters
    ----------
        cll: BaseClassifiersList
        cfg: BaseConfiguration
        pb: float = 0.5

    Returns
    -------
    BaseClassifier
    """
    rand = RandomNumberGenerator.random()
    if rand < pb:
        return choose_random_classifiers(cll, cfg)
    elif rand < pb + (1. - pb)/2.: #pb+ (1. - pb)/2. with 2 being the number of biases
        return choose_action_from_knowledge_array(cll, cfg)
    else:
        return choose_latest_action(cll, cfg)


def choose_latest_action(
        cll: BaseClassifiersList,
        cfg: BaseConfiguration
    ) -> BaseClassifier:
    """
    Computes latest executed action ("action delay bias") and return 
    a corresponding classifier.

    Parameters
    ----------
        cll: BaseClassifiersList
        cfg: BaseConfiguration

    Returns
    -------
    BaseClassifier
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
                return BaseClassifier(action=action, cfg=cfg)
        return last_executed_cls
    return choose_random_classifiers(cll, cfg)


def choose_action_from_knowledge_array(
        cll: BaseClassifiersList,
        cfg: BaseConfiguration
    ) -> BaseClassifier:
    """
    Creates 'knowledge array' that represents the average quality of the
    anticipation for each action in the current list. Chosen is
    the action, the system knows least about the consequences.
    Then a classifier that corresponds to this action is randomly returned.

    Parameters
    ----------
        cll: BaseClassifiersList
        cfg: BaseConfiguration

    Returns
    -------
    BaseClassifier
    """
    knowledge_array = {i: 0.0 for i in range(cfg.number_of_possible_actions)}
    if len(cll) > 0:
        #Sort and build knowledge array
        cll.sort(key=lambda cl: cl.action)
        for _action, _clss in groupby(cll, lambda cl: cl.action):
            _classifiers = [cl for cl in _clss]
            agg_q = sum(cl.q * cl.num for cl in _classifiers)
            agg_num = sum(cl.num for cl in _classifiers)
            knowledge_array[_action] = agg_q / float(agg_num)
        by_quality = sorted(knowledge_array.items(), key=lambda el: el[1])
        #Find action and retrun one related classifier
        action = by_quality[0][0]
        classifiers_that_match_action = [cl for cl in cll if cl.action == action]
        if len(classifiers_that_match_action) > 0:
            return RandomNumberGenerator.choice(classifiers_that_match_action)
        else:
            return BaseClassifier(action=action, cfg=cfg)
    return choose_random_classifiers(cll, cfg)


def choose_random_classifiers(
        cll: BaseClassifiersList,
        cfg: BaseConfiguration
    ) -> BaseClassifier:
    """
    Chooses one of the possible actions in the environment randomly 
    and return a corresponding classifier.

    Parameters
    ----------
        cll: BaseClassifiersList
        cfg: BaseConfiguration

    Returns
    -------
    BaseClassifier
    """
    nb_of_cll = len(cll)
    rand = RandomNumberGenerator.integers(nb_of_cll + cfg.number_of_possible_actions)
    if rand < nb_of_cll:
        return cll[rand]
    action = rand - nb_of_cll
    return BaseClassifier(action=action, cfg=cfg)


def choose_fittest_classifier(
        cll: BaseClassifiersList,
        cfg: BaseConfiguration
    ) -> BaseClassifier:
    """
    Chooses the fittest classifier in the matching set

    Parameters
    ----------
        cll: BaseClassifiersList
        cfg: BaseConfiguration

    Returns
    -------
    BaseClassifier
    """
    if len(cll) > 0:
        anticipated_change = [cl for cl in cll if cl.does_anticipate_change()]
        if len(anticipated_change) > 0:
            return max(anticipated_change, key=lambda cl: cl.fitness)
        else:
            return max(cll, key=lambda cl: cl.fitness)
    return choose_random_classifiers(cll, cfg)
