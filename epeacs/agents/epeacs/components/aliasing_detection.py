"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from epeacs import Perception
from epeacs.agents.epeacs import ClassifiersList, Condition, Configuration, PMark


def is_state_aliased(
        condition: Condition,
        mark: PMark,
        p0: Perception
    ) -> bool:
    """
    Check if the specification of unchanging components succeed or failed.
    Help to detect aliased states in POMDPs.

    Parameters
    ----------
    condition
    mark
    p0

    Returns
    -------
    bool
    """
    if mark.one_situation_in_mark():
        situation = Condition(condition)
        for idx, item in enumerate(condition):
            if item == condition.wildcard:
                situation[idx] = ''.join(str(s) for s in mark[idx])
        return situation.does_match(p0)
    return False


def should_pai_detection_apply(
        match_set, 
        time: int, 
        theta_bseq: int
    ) -> bool:
    """
    Checks the average last PAI detection to determine if a new detection is needed.
    If no classifier is in the current set, no detection is applied!

    Parameters
    ----------
    match_set
        population of classifiers (with `num` and `tbseq` properties)
    time: int
        current epoch
    theta_bseq: int
        The pai detection threshold (θga ∈ N) controls the PAI detection frequency.

    Returns
    -------
    bool
        True if pai detection should be applied, False otherwise
    """
    if match_set is None:
        return False

    overall_time = sum(cl.tbseq * cl.num for cl in match_set)
    overall_num = sum(cl.num for cl in match_set)

    if overall_num == 0:
        return False

    if time - (overall_time / overall_num) > theta_bseq:
        return True

    return False


def set_pai_detection_timestamps(match_set, epoch: int) -> None:
    """
    Sets the pai detection time stamps to the current time to control
    the detection frequency.
    Each classifier `tbseq` property in population is updated with current
    epoch

    Parameters
    ----------
    match_set
        population of classifiers
    epoch: int
        current epoch
    """
    for cl in match_set:
        cl.tbseq = epoch

def is_perceptual_aliasing_state(
        match_set: ClassifiersList,
        p0: Perception,
        cfg: Configuration
    ) -> bool:
    """
    Check if the system suffers from the perceptual aliasing issue in the penultimate perception

    Parameters
    ----------
    match_set
    p0
    cfg

    Returns
    -------
    bool
    """
    nbr_of_actions = cfg.number_of_possible_actions
    nbr_of_expected_transitions = nbr_of_actions
    most_experienced_classifiers = {}
    #Find the most experienced classifiers for all actions
    for cl in match_set:
        if cl.action not in most_experienced_classifiers.keys():
            most_experienced_classifiers[cl.action] = cl
        elif cl.exp * cl.num > most_experienced_classifiers[cl.action].exp * most_experienced_classifiers[cl.action].num:
            most_experienced_classifiers[cl.action] = cl
    #Check that all these classifiers are enough experienced
    for i in range(nbr_of_actions):
        if most_experienced_classifiers[i].exp <= cfg.theta_exp :
            return False
    #Approximate the number of expected transitions and the reachable states
    reachable_states = {}
    for i in range(nbr_of_actions):
        if not most_experienced_classifiers[i].effect.detailled_counter:
            if not most_experienced_classifiers[i].does_anticipate_change():
                nbr_of_expected_transitions -= 1
            else:
                anticipation = list(p0)
                for idx, ei in enumerate(most_experienced_classifiers[i].effect):
                    if ei != most_experienced_classifiers[i].effect.wildcard:
                        anticipation[idx] = ei
                reachable_states.update( {tuple(anticipation): 0} )
        else:
            reachable_states.update(most_experienced_classifiers[i].effect.detailled_counter)
            most_anticipated_state = max(most_experienced_classifiers[i].effect.detailled_counter.items(), key=lambda x : x[1])
            if p0 == most_anticipated_state[0]:
                nbr_of_expected_transitions -= 1
    #Compute the number of reachables states
    nbr_of_reachable_states = len(reachable_states)
    if p0 in reachable_states.keys():
        nbr_of_reachable_states -= 1
    #Compare the number of expected transitions to the number of reachable states
    if nbr_of_reachable_states > nbr_of_expected_transitions:
        return True
    return False