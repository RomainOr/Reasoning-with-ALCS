"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from bacs import Perception
from bacs.agents.bacs.classifier_components import PMark
from bacs.agents.bacs.classifier_components import Condition


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