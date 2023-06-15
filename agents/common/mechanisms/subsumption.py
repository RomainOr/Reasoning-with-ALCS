"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""


def does_subsume(
        cl,
        other_cl
    ) -> bool:
    """
    Returns if a classifier `cl` subsumes `other_cl` classifier when a classifier has to be added to a set of classifiers.

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