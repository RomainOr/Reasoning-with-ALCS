"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import random
from typing import Callable, Dict

import numpy as np

from bacs import Perception
from bacs.agents.bacs.components.subsumption_bacs import find_subsumers


def should_apply(
        action_set, 
        time: int, 
        theta_ga: int
    ) -> bool:
    """
    Checks the average last GA application to determine if a GA
    should be applied.
    If no classifier is in the current set, no GA is applied!

    Parameters
    ----------
    action_set
        population of classifiers (with `num` and `tga` properties)
    time: int
        current epoch
    theta_ga: int
        The GA application threshold (θga ∈ N) controls the GA frequency. A GA
        is applied in an action set if the average delay of the last GA
        application of the classifiers in the set is greater than θga.

    Returns
    -------
    bool
        True if GA should be applied, False otherwise
    """
    if action_set is None:
        return False

    overall_time = sum(cl.tga * cl.num for cl in action_set)
    overall_num = sum(cl.num for cl in action_set)

    if overall_num == 0:
        return False

    if time - (overall_time / overall_num) > theta_ga:
        return True

    return False


def set_timestamps(action_set, epoch: int) -> None:
    """
    Sets the GA time stamps to the current time to control
    the GA application frequency.
    Each classifier `tga` property in population is updated with current
    epoch

    Parameters
    ----------
    action_set
        population of classifiers
    epoch: int
        current epoch
    """
    for cl in action_set:
        cl.tga = epoch


def roulette_wheel_selection(population, fitnessfunc: Callable):
    """
    Select two objects from population according
    to roulette-wheel selection.

    Parameters
    ----------
    population
        population of classifiers
    fitnessfunc: Callable
        function evaluating fitness for each classifier. Very often cl.q^3
    Returns
    -------
    tuple
        two classifiers selected as parents
    """
    choices = {cl: fitnessfunc(cl) for cl in population}

    parent1 = _weighted_random_choice(choices)
    parent2 = _weighted_random_choice(choices)

    return parent1, parent2


def generalizing_mutation(cl, mu: float) -> None:
    """
    Executes the generalizing mutation in the classifier.
    Specified attributes in classifier conditions are randomly
    generalized with `mu` probability.
    """
    for idx, cond in enumerate(cl.condition):
        if cond != cl.cfg.classifier_wildcard and random.random() < mu:
            cl.condition.generalize(idx)


def two_point_crossover(parent, donor) -> None:
    """
    Executes two-point crossover using condition parts of two classifiers.
    Condition in both classifiers are changed.

    Parameters
    ----------
    parent
        Classifier
    donor
        Classifier
    """
    left, right = sorted(np.random.choice(
        range(0, parent.cfg.classifier_length + 1), 2, replace=False))

    assert left < right

    # Extract chromosomes from condition parts
    chromosome1 = parent.condition[left:right]
    chromosome2 = donor.condition[left:right]

    # Flip them
    for idx, el in enumerate(range(left, right)):
        parent.condition[el] = chromosome2[idx]
        donor.condition[el] = chromosome1[idx]


def add_classifier(
        cl, 
        p: Perception,
        population, 
        match_set, 
        action_set,
        theta_exp: int
    )-> None:
    """
    Find subsumer/similar classifier, if present - increase its numerosity,
    else add this new classifier

    Parameters
    ----------
    cl:
        newly created classifier
    p: Perception
        current perception
    population:
        population of classifiers
    match_set:
        match set
    action_set:
        action set
    theta_exp: int
        subsumption experience threshold
    """
    # Find_subsumers computes subsumer or classifier that are equal
    subsumers = find_subsumers(cl, population, theta_exp)
    # Check if subsumers exist, meaning that old_cl is mandatory not None
    if len(subsumers) == 0:
        population.append(cl)
        action_set.append(cl)
        if match_set is not None and cl.condition.does_match(p):
            match_set.append(cl)
    else:
        old_cl = subsumers[0]
        if not old_cl.is_marked():
            old_cl.num += 1


def delete_classifiers(population, match_set, action_set,
                       insize: int, theta_as: int):
    """
    Make room for new classifiers

    Parameters
    ----------
    population:
    match_set:
    action_set:
    insize: int
        number of children that will be inserted
    theta_as: int
        The action set size threshold (θas ∈ N) specifies
        the maximal number of classifiers in an action set.
    """
    while (insize + sum(cl.num for cl in action_set)) > theta_as:
        cl_del = None

        while cl_del is None:  # We must delete at least one
            set_to_iterate = [cl for cl in action_set.expand()]
            for cl in set_to_iterate:
                if random.random() < .3:
                    if cl_del is None:
                        cl_del = cl
                    else:
                        if _is_preferred_to_delete(cl_del, cl):
                            cl_del = cl

        if cl_del is not None:
            if cl_del.num > 1:
                cl_del.num -= 1
            else:
                # Removes classifier from population, match set
                # and current list
                lists = [x for x in [population, match_set, action_set] if x]
                for lst in lists:
                    lst.safe_remove(cl_del)


def _is_preferred_to_delete(cl_del, cl) -> bool:
    """
    Compares two classifiers `cl_del` (marked for deletion) with `cl` to
    check whether `cl` should be deleted instead.

    Parameters
    ----------
    cl_del: Classifier marked for deletion
    cl: Examined classifier

    Returns
    -------
    bool
        True if `cl` is "worse" than `cl_del`. False otherwise
    """
    if cl.q - cl_del.q < -0.1:
        return True

    if abs(cl.q - cl_del.q) <= 0.1:
        if cl.is_marked() and not cl_del.is_marked():
            return True
        elif cl.is_marked() or not cl_del.is_marked():
            if cl.tav > cl_del.tav:
                return True

    return False


def _weighted_random_choice(choices: Dict):
    max = sum(choices.values())
    pick = random.uniform(0, max)
    current = 0

    for key, value in choices.items():
        current += value
        if current > pick:
            return key
