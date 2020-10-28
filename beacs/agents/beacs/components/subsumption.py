"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import List


def find_subsumers(
        cl,
        population,
        theta_exp: int
    ) -> List:
    """
    Looks for subsumers of `cl` inside `population`.

    Parameters
    ----------
    cl:
        Classifier
    population:
        Population of classifiers
    theta_exp: int
        Subsumption experience threshold

    Returns
    -------
    List
        List of subsumers (classifiers) sorted by specificity (most general
        are first)
    """
    subsumers = [sub for sub in population if does_subsume(sub, cl, theta_exp)]
    return sorted(subsumers, key=lambda cl: cl.condition.specificity)


def does_subsume(
        cl,
        other_cl,
        theta_exp: int
    ) -> bool:
    """
    Returns if a classifier `cl` subsumes `other_cl` classifier.
    No need to check condition when does_subsume is only applied 
    on the matching set or the action set.

    Parameters
    ----------
    cl:
        Subsumer classifier
    other_cl:
        Other classifier
    theta_exp: int
        Experience threshold

    Returns
    -------
    bool
        True if `other_cl` classifier is subsumed
    """
    if is_subsumer(cl, theta_exp) and \
        cl.is_more_general(other_cl) and \
            cl.action == other_cl.action and \
                cl.behavioral_sequence == other_cl.behavioral_sequence and \
                    cl.effect.subsumes(other_cl.effect):
        return True

    return False


def is_subsumer(
        cl,
        theta_exp: int
    ) -> bool:
    """
    Determines whether the classifier satisfies the subsumer criteria.

    Parameters
    ----------
    cl:
        Classifier
    theta_exp: int
        Experience threshold to be considered as experienced

    Returns
    -------
    bool
        True is classifier can be considered as subsumer
    """
    if cl.exp > theta_exp:
        if cl.is_reliable():
            if not cl.is_marked():
                return True

    return False
