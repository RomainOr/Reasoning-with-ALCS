"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import random
from typing import Callable, Dict

from beacs import Perception, UBR
from beacs.agents.beacs.components.subsumption import find_subsumers


def should_apply(
        action_set, 
        time: int, 
        theta_rc: int
    ) -> bool:
    """
    Checks the average last RC application to determine if a RC
    should be applied.
    If no classifier is in the current set, no RC is applied!

    Parameters
    ----------
    action_set
        Population of classifiers (with `num` and `trc` properties)
    time: int
        Current epoch
    theta_rc: int
        The RC application threshold (θrc ∈ N) controls the RC frequency.
        A RC is applied in an action set if the average delay of the last RC
        application of the classifiers in the set is greater than θrc.

    Returns
    -------
    bool
        True if RC should be applied, False otherwise
    """
    if action_set is None or not action_set:
        return False

    overall_time = sum(cl.trc * cl.num for cl in action_set)
    overall_num = sum(cl.num for cl in action_set)

    if overall_num == 0:
        return False

    if time - (overall_time / overall_num) > theta_rc:
        return True

    return False


def set_timestamps(
        action_set,
        epoch: int
    ) -> None:
    """
    Sets the RC time stamps to the current time to control
    the RC application frequency.
    Each classifier `trc` property in population is updated with current
    epoch

    Parameters
    ----------
    action_set
        Population of classifiers
    epoch: int
        Current epoch
    """
    for cl in action_set:
        cl.trc = epoch


def roulette_wheel_selection(
        population,
        fitnessfunc: Callable
    ) -> tuple:
    """
    Selects two objects from population according
    to roulette-wheel selection.

    Parameters
    ----------
    population
        Population of classifiers
    fitnessfunc: Callable
        Function evaluating fitness for each classifier.

    Returns
    -------
    tuple
        Two classifiers selected as parents
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


def rule_combination(
        cl1,
        cl2
    ) -> None:
    """
    Executes a particular mutation in the behavioral classifiers.

    Parameters
    ----------
    cl1
        First behavioral classifier
    cl2
        Second behavioral classifier
    """
    for idx in range(cl1.cfg.classifier_length):
        # Both attributes are generalized, skip
        if cl1.condition[idx] == cl1.cfg.classifier_wildcard and \
            cl2.condition[idx] == cl2.cfg.classifier_wildcard :
            if cl1.anticipation[0][idx] != cl1.cfg.classifier_wildcard and cl2.anticipation[0][idx] != cl2.cfg.classifier_wildcard and cl1.anticipation[0][idx].does_intersect_with(cl2.anticipation[0][idx]):
                cl1.anticipation[0][idx].widen_with_ubr(cl2.anticipation[0][idx])
                cl2.anticipation[0][idx].widen_with_ubr(cl1.anticipation[0][idx])
                continue
            continue
        # One attribute is generalized
        if cl1.condition[idx] != cl1.cfg.classifier_wildcard and \
            cl2.condition[idx] == cl2.cfg.classifier_wildcard:
            if cl1.anticipation[0][idx] == cl1.cfg.classifier_wildcard and cl2.anticipation[0][idx] != cl2.cfg.classifier_wildcard and cl2.anticipation[0][idx].does_intersect_with(cl1.condition[idx]):
                cl1.anticipation[0][idx] = UBR.copy(cl1.condition[idx])
                cl1.anticipation[0][idx].widen_with_ubr(cl2.anticipation[0][idx])
                continue
            if cl1.anticipation[0][idx] != cl1.cfg.classifier_wildcard and cl2.anticipation[0][idx] != cl2.cfg.classifier_wildcard and cl1.anticipation[0][idx].does_intersect_with(cl2.anticipation[0][idx]):
                cl1.anticipation[0][idx].widen_with_ubr(cl2.anticipation[0][idx])
                cl2.anticipation[0][idx].widen_with_ubr(cl1.anticipation[0][idx])
                continue
            continue
        # One attribute is generalized
        if cl1.condition[idx] == cl1.cfg.classifier_wildcard and \
            cl2.condition[idx] != cl2.cfg.classifier_wildcard:
            if cl1.anticipation[0][idx] != cl1.cfg.classifier_wildcard and cl2.anticipation[0][idx] == cl2.cfg.classifier_wildcard and cl1.anticipation[0][idx].does_intersect_with(cl2.condition[idx]):
                cl2.anticipation[0][idx] = UBR.copy(cl2.condition[idx])
                cl2.anticipation[0][idx].widen_with_ubr(cl1.anticipation[0][idx])
                continue
            if cl1.anticipation[0][idx] != cl1.cfg.classifier_wildcard and cl2.anticipation[0][idx] != cl2.cfg.classifier_wildcard and cl1.anticipation[0][idx].does_intersect_with(cl2.anticipation[0][idx]):
                cl1.anticipation[0][idx].widen_with_ubr(cl2.anticipation[0][idx])
                cl2.anticipation[0][idx].widen_with_ubr(cl1.anticipation[0][idx])
                continue
            continue
        if cl1.condition[idx].does_intersect_with(cl2.condition[idx]):
            if cl1.anticipation[0][idx] == cl1.cfg.classifier_wildcard and cl2.anticipation[0][idx] != cl2.cfg.classifier_wildcard and cl2.anticipation[0][idx].does_intersect_with(cl1.condition[idx]):
                cl1.anticipation[0][idx] = UBR.copy(cl1.condition[idx])
                cl1.anticipation[0][idx].widen_with_ubr(cl2.anticipation[0][idx])
                continue
            if cl1.anticipation[0][idx] != cl1.cfg.classifier_wildcard and cl2.anticipation[0][idx] == cl2.cfg.classifier_wildcard and cl1.anticipation[0][idx].does_intersect_with(cl2.condition[idx]):
                cl2.anticipation[0][idx] = UBR.copy(cl2.condition[idx])
                cl2.anticipation[0][idx].widen_with_ubr(cl1.anticipation[0][idx])
                continue
            if cl1.anticipation[0][idx] != cl1.cfg.classifier_wildcard and cl2.anticipation[0][idx] != cl2.cfg.classifier_wildcard and cl1.anticipation[0][idx].does_intersect_with(cl2.anticipation[0][idx]):
                cl1.anticipation[0][idx].widen_with_ubr(cl2.anticipation[0][idx])
                cl2.anticipation[0][idx].widen_with_ubr(cl1.anticipation[0][idx])
                continue

def add_classifier(
        cl, 
        p: Perception,
        population, 
        match_set, 
        action_set,
        theta_exp: int
    )-> None:
    """
    Finds subsumer/similar classifier, if present - increase its numerosity,
    else add this new classifier

    Parameters
    ----------
    cl:
        Newly created classifier
    p: Perception
        Current perception
    population:
        Population of classifiers
    match_set:
        Match set
    action_set:
        Action set
    theta_exp: int
        Subsumption experience threshold
    """
    # Find_subsumers computes subsumer or classifier that are equal
    subsumers = find_subsumers(cl, action_set, theta_exp)
    # Check if subsumers exist, meaning that old_cl is mandatory not None
    if len(subsumers) == 0:
        old_cl = next(filter(lambda other: cl == other, action_set), None)
        if old_cl:
            if not old_cl.is_marked():
                old_cl.num += 1
        else:
            population.append(cl)
            action_set.append(cl)
            if match_set is not None and cl.does_match(p):
                match_set.append(cl)
    else:
        old_cl = subsumers[0]
        if not old_cl.is_marked():
            old_cl.num += 1


def delete_classifiers(
        population,
        match_set,
        action_set,
        insize: int, 
        theta_as: int
    ) -> None:
    """
    Makes room for new classifiers

    Parameters
    ----------
    population:
        Whole population of classifiers
    match_set:
        Population of classifiers that match p0
    action_set:
        Population of classifiers from the matching that have the selection action
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


def _is_preferred_to_delete(
        cl_del,
        cl
    ) -> bool:
    """
    Compares two classifiers `cl_del` (marked for deletion) with `cl` to
    check whether `cl` should be deleted instead.

    Parameters
    ----------
    cl_del
        Classifier marked for deletion
    cl
        Examined classifier

    Returns
    -------
    bool
        True if `cl` is "worse" than `cl_del`. False otherwise
    """
    # TODO : Inductive principles to check
    if cl.q - cl_del.q < -0.1:
        return True

    if abs(cl.q - cl_del.q) <= 0.1:
        if cl.is_marked() and not cl_del.is_marked():
            return True
        elif cl.is_marked() or not cl_del.is_marked():
            if cl.tav > cl_del.tav:
                return True

    return False
