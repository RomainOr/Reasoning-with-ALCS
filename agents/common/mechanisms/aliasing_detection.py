"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common import Perception
from agents.common.classifier_components import Condition, PMark


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