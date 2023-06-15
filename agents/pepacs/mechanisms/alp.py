"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Optional

from agents.common.Perception import Perception
import agents.common.mechanisms.aliasing_detection as aliasing_detection
import agents.common.mechanisms.alp as alp_common

from agents.pepacs.classifier_components import PEPACSClassifier


def expected_case(
        cl: PEPACSClassifier,
        p0: Perception,
        p1: Perception,
        time: int
    ) -> Optional[PEPACSClassifier]:
    """
    Controls the expected case of a classifier with the help of 
    Specification of Unchanging Components.

    Parameters
    ----------
    cl
    p0
    p1
    time

    Returns
    ----------
    New classifier or None
    """
    if cl.is_enhanced():
        cl.effect.update_enhanced_effect_probs(p1, cl.cfg.beta_pep)

    if aliasing_detection.is_state_aliased(cl.condition, cl.mark, p0):
        cl.ee = True

    return alp_common.expected_case(cl, p0, time, p1)