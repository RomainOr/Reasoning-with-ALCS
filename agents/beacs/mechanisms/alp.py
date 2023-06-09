"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Optional

import agents.common.mechanisms.aliasing_detection as aliasing_detection
import agents.common.mechanisms.alp as alp_common
from agents.common.BaseConfiguration import BaseConfiguration
from agents.common.Perception import Perception

import agents.beacs.mechanisms.pai_detection as pai_detection
from agents.beacs.classifier_components.BEACSClassifier import BEACSClassifier
from agents.beacs.mechanisms.build_behavioral_sequences import create_behavioral_classifier


def expected_case(
        cl: BEACSClassifier,
        p0: Perception,
        time: int
    ):
    """
    Controls the expected case of a classifier with the help of 
    Specification of Unchanging Components.

    Returns
    ----------
    Bool related to aliasing, New classifier or None
    """
    is_aliasing_detected = False

    if cl.is_enhanced() and cl.aliased_state == p0:
        is_aliasing_detected = True

    if aliasing_detection.is_state_aliased(cl.condition, cl.mark, p0):
        cl.ee = True
        is_aliasing_detected = True

    if cl.is_enhanced():
        diff = cl.mark.get_differences(cl.aliased_state)
    else:
        diff = cl.mark.get_differences(p0)
    if diff.specificity == 0:
        cl.increase_quality()
        return is_aliasing_detected, None

    return is_aliasing_detected, alp_common.specification_unchanging_components(cl, diff, time)

    
def apply_enhanced_effect_part_check(
        action_set,
        new_list,
        p0: Perception,
        time: int
    ) -> None:
    """
    Used to build enhanced classifiers
    """
    candidates = [cl for cl in action_set if cl.ee]
    if len(candidates) < 2:
        return
    for i, cl1 in enumerate(candidates):
        for cl2 in candidates[i:]:
            if cl1.mark == cl2.mark and \
            not cl1.effect.subsumes(cl2.effect) and \
            not cl2.effect.subsumes(cl1.effect) and \
            (cl1.aliased_state == Perception.empty() or cl1.aliased_state == p0) and \
            (cl2.aliased_state == Perception.empty() or cl2.aliased_state == p0):
                new_classifier = cl1.merge_with(cl2, p0, time)
                alp_common.add_classifier(new_classifier, action_set, new_list)
                break


def apply_perceptual_aliasing_issue_management(
        population,
        t_2_match_set,
        t_1_match_set,
        match_set,
        action_set,
        penultimate_classifier: BEACSClassifier,
        potential_cls_for_pai,
        new_list,
        p0: Perception,
        p1: Perception,
        time: int,
        pai_states_memory,
        cfg: BaseConfiguration
    ) -> None:
    """
    Used to manage the detection of PAI and to manage the behavioral classifiers
    """
    # First, try to detect if it is time to detect a pai state - no need to compute this every time
    knowledge_from_match_set = [cl for cl in t_1_match_set if
        cl.behavioral_sequence is None and
        (not cl.is_marked() or cl.mark.corresponds_to(p0)) and 
        (cl.aliased_state == Perception.empty() or cl.aliased_state == p0)
    ]
    if pai_detection.should_pai_detection_apply(knowledge_from_match_set, time, cfg.theta_bseq):
        # We set the related timestamp t_bseq of the classifiers in the match set
        pai_detection.set_pai_detection_timestamps(knowledge_from_match_set, time)
        # We check we have enough information from classifiers in the matching set to do the detection
        enough_information, most_experienced_classifiers = pai_detection.enough_information_to_try_PAI_detection(knowledge_from_match_set, cfg)
        if enough_information:
        # The system tries to determine is it suffers from the perceptual aliasing issue
            if pai_detection.is_perceptual_aliasing_state(most_experienced_classifiers, p0, cfg) > 0:
                # Add if needed the new pai state in memory
                if p0 not in pai_states_memory:
                    pai_states_memory.append(p0)
            else:
                # Remove if needed the pai state from memory and delete all behavioral classifiers created for this state
                if p0 in pai_states_memory:
                    pai_states_memory.remove(p0)
                    behavioral_classifiers_to_delete = [cl for cl in population if cl.pai_state == p0]
                    for cl in behavioral_classifiers_to_delete:
                        lists = [x for x in [population, match_set, action_set] if x]
                        for lst in lists:
                            lst.safe_remove(cl)

    # Create new behavioral classifiers
    if p0 in pai_states_memory and len(potential_cls_for_pai) > 0:
        for candidate in potential_cls_for_pai:
            new_cl = create_behavioral_classifier(penultimate_classifier, candidate, p1, p0, time)
            if new_cl:
                alp_common.add_classifier(new_cl, t_2_match_set, new_list)