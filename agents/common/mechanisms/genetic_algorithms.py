"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Callable, Dict

from agents.common.BaseClassifiersList import BaseClassifiersList
from agents.common.BaseConfiguration import BaseConfiguration
from agents.common.Perception import Perception
from agents.common.RandomNumberGenerator import RandomNumberGenerator
from agents.common.classifier_components.BaseClassifier import BaseClassifier
from agents.common.mechanisms.subsumption import does_subsume


def should_apply(
        action_set: BaseClassifiersList, 
        time: int, 
        theta_ga: int
    ) -> bool:
    """
    Checks the average last GA application to determine if a GA
    should be applied.
    If no classifier is in the current set, no GA is applied!

    Parameters
    ----------
        action_set: BaseClassifiersList
        time: int
        theta_ga: int

    Returns
    -------
    bool
    """
    if action_set is None or not action_set:
        return False

    overall_time = sum(cl.tga * cl.num for cl in action_set)
    overall_num = sum(cl.num for cl in action_set)

    if overall_num == 0:
        return False

    if time - (overall_time / overall_num) > theta_ga:
        return True

    return False


def set_timestamps(
        action_set: BaseClassifiersList,
        epoch: int
    ) -> None:
    """
    Sets the GA time stamps to the current time to control
    the GA application frequency.
    Each classifier `tga` property in population is updated with current
    epoch

    Parameters
    ----------
        action_set: BaseClassifiersList
        epoch: int
    """
    for cl in action_set:
        cl.tga = epoch


def roulette_wheel_selection(
        population: BaseClassifiersList,
        fitnessfunc: Callable
    ) -> tuple:
    """
    Selects two objects from population according
    to roulette-wheel selection.

    Parameters
    ----------
        population: BaseClassifiersList
        fitnessfunc: Callable

    Returns
    -------
    tuple
    """
    def _weighted_random_choice(choices: Dict):
        maximum = sum(choices.values())
        pick = RandomNumberGenerator.uniform(0, maximum)
        current = 0
        for key, value in choices.items():
            current += value
            if current > pick:
                return key

    choices = {cl: fitnessfunc(cl) for cl in population}
    parent1 = _weighted_random_choice(choices)
    parent2 = _weighted_random_choice(choices)
    return parent1, parent2


def mutation(
        cl1: BaseClassifier,
        cl2: BaseClassifier,
        mu: float
    ) -> None:
    """
    Executes mutation in one classifier.
    Specified attributes in classifier conditions are randomly
    generalized with `mu` probability.

    Parameters
    ----------
        cl1: BaseClassifier
        cl2: BaseClassifier
        mu: float
    """
    for idx in range(len(cl1.condition)):
        if cl1.condition[idx] != cl1.cfg.classifier_wildcard and RandomNumberGenerator.random() < mu:
            cl1.condition.generalize(idx)
        if cl2.condition[idx] != cl2.cfg.classifier_wildcard and RandomNumberGenerator.random() < mu:
            cl2.condition.generalize(idx)


def two_point_crossover(
        parent: BaseClassifier,
        donor: BaseClassifier
    ) -> None:
    """
    Executes two-point crossover using condition parts of two classifiers.
    Condition in both classifiers are changed.

    Parameters
    ----------
        parent: BaseClassifier
        donor: BaseClassifier
    """
    left, right = sorted(RandomNumberGenerator.choice(
        range(0, parent.cfg.classifier_length + 1), 2, replace=False))

    # Extract chromosomes from condition parts
    chromosome1 = parent.condition[left:right]
    chromosome2 = donor.condition[left:right]

    # Flip them
    for idx, el in enumerate(range(left, right)):
        parent.condition[el] = chromosome2[idx]
        donor.condition[el] = chromosome1[idx]


def delete_classifiers(
        population: BaseClassifiersList,
        match_set: BaseClassifiersList,
        action_set: BaseClassifiersList,
        insize: int, 
        theta_as: int
    ) -> None:
    """
    Makes room for new classifiers

    Parameters
    ----------
        population: BaseClassifiersList
        match_set: BaseClassifiersList
        action_set: BaseClassifiersList
        insize: int
        theta_as: int
    """
    while (insize + sum(cl.num for cl in action_set)) > theta_as: 
        # We must delete at least one
        set_to_iterate = [cl for cl in action_set.expand()]
        cl_del = RandomNumberGenerator.choice(set_to_iterate)
        for cl in set_to_iterate:
            if RandomNumberGenerator.random() < .3:
                if _is_preferred_to_delete(cl_del, cl):
                    cl_del = cl
        if cl_del.num > 1:
            cl_del.num -= 1
        else:
            # Removes classifier from population, match set
            # and current list
            lists = [x for x in [population, match_set, action_set] if x]
            for lst in lists:
                lst.safe_remove(cl_del)


def _is_preferred_to_delete(
        cl_del: BaseClassifier,
        cl: BaseClassifier
    ) -> bool:
    """
    Compares two classifiers `cl_del` (marked for deletion) with `cl` to
    check whether `cl` should be deleted instead.

    Parameters
    ----------
        cl_del: BaseClassifier
        cl: BaseClassifier

    Returns
    -------
    bool
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
        child: BaseClassifier, 
        population: BaseClassifiersList,
        new_list: list
    )-> None:
    """
    Looks for subsuming / similar classifiers in the population of classifiers
    and those created in the current GA run.

    If a similar classifier was found it's numerosity is increased,
    otherwise `child_cl` is added to `new_list`.

    Parameters
    ----------
        child: BaseClassifier
        population: BaseClassifiersList
        new_list: list
    """
    old_cl = None
    equal_cl = None
    # Look if there is a classifier that subsumes the insertion candidate
    for cl in population:
        if does_subsume(cl, child):
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
            old_cl.num += child.num


def apply(
        cls_ClassifiersList: BaseClassifiersList,
        mutate_function: Callable,
        crossover_function: Callable,
        population: BaseClassifiersList,
        match_set: BaseClassifiersList,
        action_set: BaseClassifiersList,
        p0: Perception,
        p1: Perception,
        time: int,
        cfg: BaseConfiguration
    ) -> None:
    """
    Apply the whole genetic generalization mechanism to the action set.

    Parameters
    ----------
        cls_ClassifiersList: BaseClassifiersList
        mutate_function: Callable
        crossover_function: Callable
        population: BaseClassifiersList
        match_set: BaseClassifiersList
        action_set: BaseClassifiersList
        p0: Perception
        p1: Perception
        time: int
        cfg: BaseConfiguration
    """

    if should_apply(action_set, time, cfg.theta_ga):
        set_timestamps(action_set, time)
        # Select parents
        parent1, parent2 = roulette_wheel_selection(
            action_set, 
            lambda cl: pow(cl.q, 3)
        )
        child1 = parent1.copy(time=time, perception=p1)
        child2 = parent2.copy(time=time, perception=p1)
        # Execute mutation
        mutate_function(child1, child2, cfg.mu)
        # Execute cross-over
        if RandomNumberGenerator.random() < cfg.chi:
            if child1.effect == child2.effect:
                crossover_function(child1, child2)
                # Update quality and reward
                child1.average_fitnesses_from_other_cl(child2)
        child1.q /= 2
        child2.q /= 2
        # We are interested only in classifiers with specialized condition
        children = {cl for cl in [child1, child2] if cl.condition.specificity > 0}
        delete_classifiers(
            population,
            match_set,
            action_set,
            len(children),
            cfg.theta_as
        )
        new_list = cls_ClassifiersList()
        # check for subsumers / similar classifiers
        for child in children:
            add_classifier(
                child,
                action_set,
                new_list
            )
        # Merge classifiers from new_list into self and population
        cls_ClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)