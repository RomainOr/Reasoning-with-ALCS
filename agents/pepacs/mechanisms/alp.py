"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from agents.beacs.BEACSClassifiersList import PEPACSClassifiersList

import agents.common.mechanisms.aliasing_detection as aliasing_detection
import agents.common.mechanisms.alp as alp_common
from agents.common.Perception import Perception
from agents.common.RandomNumberGenerator import RandomNumberGenerator

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
        cl: PEPACSClassifier
        p0: Perception
        p1: Perception
        time: int

    Returns
    ----------
    New classifier or None
    """
    if cl.is_enhanced():
        cl.effect.update_enhanced_effect_probs(p1, cl.cfg.beta_pep)

    if aliasing_detection.is_state_aliased(cl.condition, cl.mark, p0):
        cl.ee = True

    return alp_common.expected_case(cl, p0, time, p1)


def apply_enhanced_effect_part_check(
        action_set: PEPACSClassifiersList,
        new_list: PEPACSClassifiersList,
        previous_situation: Perception,
        time: int
    ) -> None:
    """
    Used to build enhanced classifiers

    Parameters
    ----------
        action_set: PEPACSClassifiersList
        new_list: PEPACSClassifiersList
        previous_situation: Perception
        time: int
    """
    # Create a list of candidates.
    # Every enhanceable classifier is a candidate.
    candidates = [cl for cl in action_set if cl.ee]
    # If there are less than 2 candidates, don't do it
    if len(candidates) < 2:
        return
    for candidate in candidates:
        candidates2 = [cl for cl in candidates if candidate != cl and cl.mark == candidate.mark]
        if len(candidates2) > 0:
            merger = RandomNumberGenerator.choice(candidates2)
            new_classifier = candidate.merge_with(merger, previous_situation, time)
            if new_classifier is not None:
                alp_common.add_classifier(new_classifier, action_set, new_list)
    return new_list