"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Optional

from agents.common.Perception import Perception
from agents.common.classifier_components.BaseClassifier import BaseClassifier
import agents.common.mechanisms.aliasing_detection as aliasing_detection
import agents.common.mechanisms.alp as alp_common

from agents.bacs.mechanisms.build_behavioral_sequences import create_behavioral_classifier


def expected_case(
        last_activated_classifier: BaseClassifier,
        cl: BaseClassifier,
        p0: Perception,
        p1: Perception,
        time: int
    ) -> Optional[BaseClassifier]:
    """
    Controls the expected case of a classifier with the help of 
    Specification of Unchanging Components.
    Controls also the case when the specification of unchanging
    components failed by creating classifiers with action chunks,
    only if the action set is not a behavioral action set.

    Parameters
    ----------
        last_activated_classifier: BaseClassifier
        cl: BaseClassifier
        p0: Perception
        p1: Perception
        time: int

    Returns
    ----------
    New classifier or None
    """

    if aliasing_detection.is_state_aliased(cl.condition, cl.mark, p0):
        if cl.cfg.bs_max != 0 and last_activated_classifier is not None:
            child = create_behavioral_classifier(last_activated_classifier, cl, time)
            if child:
                return child

    return alp_common.expected_case(cl, p0, time, p1)