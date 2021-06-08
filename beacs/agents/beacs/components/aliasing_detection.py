"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from beacs import Perception
from beacs.agents.beacs import ClassifiersList, Condition, Configuration, Effect, PMark


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
    condition: Condition
        Condition of the classifier used to check if the state is aliased
    mark: PMark
        Mark of the classifier used to check if the state is aliased
    p0: Perception
        Perception of the state in the previous step

    Returns
    -------
    bool
        True if the state p0 is aliased
    """
    if mark.one_situation_in_mark():
        situation = Condition(condition)
        for idx, item in enumerate(condition):
            if item == condition.wildcard:
                situation[idx] = ''.join(str(s) for s in mark[idx])
        return situation.does_match(p0)
    return False


def should_pai_detection_apply(
        match_set: ClassifiersList,
        time: int, 
        theta_bseq: int
    ) -> bool:
    """
    Checks the average last PAI detection to determine if a new detection is needed.
    If no classifier is in the current set, no detection is applied!

    Parameters
    ----------
    match_set: ClassifiersList
        Population of classifiers having no mark or a mark that corresponds to p0
        and whose condition matches p0, without the behavioral ones.
    time: int
        Current epoch
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


def set_pai_detection_timestamps(
        match_set, 
        epoch: int
    ) -> None:
    """
    Sets the pai detection time stamps to the current time to control
    the detection frequency.
    Each classifier `tbseq` property in population is updated with current
    epoch

    Parameters
    ----------
    match_set: ClassifiersList
        Population of classifiers having no mark or a mark that corresponds to p0
        and whose condition matches p0, without the behavioral ones.
    epoch: int
        Current epoch
    """
    for cl in match_set:
        cl.tbseq = epoch


def enough_information_to_try_PAI_detection(
        match_set: ClassifiersList,
        cfg: Configuration
    ):
    """
    Check if we can collect enough information from the match set to try to detect if the state related to the match set is a PAI state

    Parameters
    ----------
    match_set: ClassifiersList
        Population of classifiers having no mark or a mark that corresponds to p0
        and whose condition matches p0, without the behavioral ones.
    cfg: Configuration
        Configuration used in the ALCS

    Returns
    -------
    bool, dict
        True if we can try to detect if the aliased state is a PAI one and the related classifiers in a dict
    """

    nbr_of_actions = cfg.number_of_possible_actions
    #Find the most experienced classifiers for all actions
    most_experienced_classifiers = {}
    for cl in match_set:
        if cl.is_reliable() and cl.is_experienced():
            if cl.action not in most_experienced_classifiers:
                most_experienced_classifiers[cl.action] = cl
            elif cl.exp * pow(cl.q, 3) > most_experienced_classifiers[cl.action].exp * pow(most_experienced_classifiers[cl.action].q, 3) :
                most_experienced_classifiers[cl.action] = cl
    #Check that each action get an associated experienced and reliable classifier
    for i in range(nbr_of_actions):
        if i not in most_experienced_classifiers:
            return False, None
    return True, most_experienced_classifiers
    

def is_perceptual_aliasing_state(
        most_experienced_classifiers: dict,
        p0: Perception,
        cfg: Configuration
    ) -> bool:
    """
    Check if the system suffers from the perceptual aliasing issue in the penultimate perception
    from the associated match set without the behavioral classifiers

    Parameters
    ----------
    classifiers: dict
        Classifiers to do the detection that are the most experienced and reliable for each action
    p0: Perception
        Perception of the state in the previous step
    cfg: Configuration
        Configuration used in the ALCS

    Returns
    -------
    bool
        True if the aliased state is associated to the Perceptual Aliasing Issue
    """

    def _build_anticipation(
            p0: Perception,
            effect: Effect
        ) -> tuple:
        """
        Build from the perception and the effect of a classifier, the complete expected anticipation

        Parameters
        ----------
        p0: Perception
            Perception of the state in the previous step
        effect: Effect
            Anticipation associated to a classifier

        Returns
        -------
        tuple
            Complete expected anticipation as a tuple
        """
        anticipation = list(p0)
        for idx, ei in enumerate(effect):
            if ei != effect.wildcard:
                anticipation[idx] = ei
        return tuple(anticipation)

    nbr_of_actions = cfg.number_of_possible_actions
    nbr_of_expected_transitions = nbr_of_actions
    #Approximate the number of expected transitions and the reachable states
    reachable_states = {}
    list_of_most_anticipated_state = []
    for i in range(nbr_of_actions):
        effect_counter_dict = { _build_anticipation(p0, effect) : counter for effect, counter in zip(most_experienced_classifiers[i].effect.effect_list, most_experienced_classifiers[i].effect.effect_detailled_counter)}
        reachable_states.update(effect_counter_dict)
        most_anticipated_state = max(effect_counter_dict.items(), key=lambda x : x[1])
        if most_anticipated_state[0] == p0:
            nbr_of_expected_transitions -= 1
            if p0 not in list_of_most_anticipated_state:
                list_of_most_anticipated_state.append(p0)
        elif most_anticipated_state[0] in list_of_most_anticipated_state:
            nbr_of_expected_transitions -= 1
        else:
            list_of_most_anticipated_state.append(most_anticipated_state[0])
    #Compute the number of reachables states
    nbr_of_reachable_states = len(reachable_states)
    if p0 in reachable_states:
        nbr_of_reachable_states -= 1
    #Compare the number of expected transitions to the number of reachable states
    if nbr_of_reachable_states > nbr_of_expected_transitions:
        return True
    return False