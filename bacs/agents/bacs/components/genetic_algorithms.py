"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import random
from typing import Callable, Dict

import numpy as np

from bacs import Perception
from bacs.agents.bacs.components.subsumption import does_subsume


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
    if action_set is None or not action_set:
        return False

    if action_set[0].behavioral_sequence:
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
    def _weighted_random_choice(choices: Dict):
        max = sum(choices.values())
        pick = random.uniform(0, max)
        current = 0
        for key, value in choices.items():
            current += value
            if current > pick:
                return key

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


def add_classifier(
        child, 
        population,
        new_list,
        theta_exp
    )-> None:
    """
    Looks for subsuming / similar classifiers in the population of classifiers
    and those created in the current GA run.

    If a similar classifier was found it's numerosity is increased,
    otherwise `child_cl` is added to `new_list`.

    Parameters
    ----------
    child:
        New classifier to examine
    population:
        List of classifiers
    new_list:
        A list of newly created classifiers in this GA run
    theta_exp:
        Experience threshold
    """
    old_cl = None
    equal_cl = None

    # Look if there is a classifier that subsumes the insertion candidate
    for cl in population:
        if does_subsume(cl, child, theta_exp):
            if old_cl is None or cl.is_more_general(old_cl):
                old_cl = cl
        elif cl == child:
            equal_cl = cl

    # Check if there is similar classifier already in the population, previously found
    if old_cl is None:
        old_cl = equal_cl

    # Check if any similar classifier was in this GA run
    if old_cl is None:
        for cl in new_list:
            if cl == child:
                old_cl = cl
                break

    if old_cl is None:
        new_list.append(child)
    else:
        if not old_cl.is_marked():
                old_cl.num += 1