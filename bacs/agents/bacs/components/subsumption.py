"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import List


def does_subsume(cl, other_cl, theta_exp: int) -> bool:
    """
    Returns if a classifier `cl` subsumes `other_cl` classifier
    No need to chech condition when does_subsume is only applied 
    on the matching set or the action set

    Parameters
    ----------
    cl:
        subsumer classifier
    other_cl:
        other classifier
    theta_exp: int
        experience threshold

    Returns
    -------
    bool
        True if `other_cl` classifier is subsumed, False otherwise
    """
    if is_subsumer(cl, theta_exp) and \
        cl.is_more_general(other_cl) and \
            cl.action == other_cl.action and \
            cl.behavioral_sequence == other_cl.behavioral_sequence and \
            cl.effect.subsumes(other_cl.effect):
        return True

    return False


def is_subsumer(cl, theta_exp: int) -> bool:
    """
    Determines whether the classifier satisfies the subsumer criteria.

    Parameters
    ----------
    cl:
        classifier
    theta_exp: int
        Experience threshold to be considered as experienced

    Returns
    -------
    bool
        True is classifier can be considered as subsumer,
        False otherwise
    """
    if cl.exp > theta_exp:
        if cl.is_reliable():
            if not cl.is_marked():
                return True

    return False
