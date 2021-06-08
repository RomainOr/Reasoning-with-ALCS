"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import List


def find_subsumers(
        cl,
        population
    ) -> List:
    """
    Looks for subsumers of `cl` inside `population`.

    Parameters
    ----------
    cl:
        Classifier
    population:
        Population of classifiers

    Returns
    -------
    List
        List of subsumers (classifiers) sorted by specificity (most general
        are first)
    """
    subsumers = [sub for sub in population if does_subsume(sub, cl)]
    return sorted(subsumers, key=lambda cl: cl.condition.specificity)


def does_subsume(
        cl,
        other_cl
    ) -> bool:
    """
    Returns if a classifier `cl` subsumes `other_cl` classifier.
    Condition is checked as this function can be applied on the whole population or in the action set.
    Called by add_classifier.py and find_subsumers() in genetic_algorithms.py.

    Parameters
    ----------
    cl:
        Subsumer classifier
    other_cl:
        Other classifier

    Returns
    -------
    bool
        True if `other_cl` classifier is subsumed
    """
    if cl.is_hard_subsumer_criteria_satisfied(other_cl) and \
        cl.is_more_general(other_cl) and \
            cl.does_match(other_cl.condition) and \
                cl.action == other_cl.action and \
                    cl.behavioral_sequence == other_cl.behavioral_sequence and \
                        cl.effect.subsumes(other_cl.effect):
        return True

    return False
