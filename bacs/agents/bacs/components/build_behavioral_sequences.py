"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Optional

from bacs import Perception
from bacs.agents.bacs import Classifier
from bacs.agents.bacs.ProbabilityEnhancedAttribute import ProbabilityEnhancedAttribute

def _updated_passthrough(
        child_effect, 
        penultimate_effect, 
        last_effect, 
        perception,
        child_condition
    ):
    """
    Passthrough operator defined by Stolzmann that we have refined.
    It is only used on the effect component of classifiers.

    Parameters
    ----------
    result
        The effect component to compute
    penultimate_classifier
    last_effect
    perception
    condition
        Condition component to remove unnecessary specification of effect attributes
    """
    for i in range(len(child_effect)):
        if last_effect[i] == child_effect.wildcard:
            if isinstance(penultimate_effect[i], ProbabilityEnhancedAttribute):
                child_effect[i] = perception[i]
            else:
                child_effect[i] = penultimate_effect[i]
        else:
            if isinstance(last_effect[i], ProbabilityEnhancedAttribute):
                child_effect[i] = perception[i]
            else:
                child_effect[i] = last_effect[i]
    # Refining effect
    for idx, effect_item in enumerate(child_effect):
        if effect_item != child_effect.wildcard and effect_item == child_condition[idx]:
            child_effect[idx] = child_effect.wildcard

def create_behavioral_classifier(
        last_activated_classifier: Classifier,
        cl: Classifier,
        p1: Perception
    ) -> Optional[Classifier]:
    """
    Build a behavioral classifier.

    Parameters
    ----------
    last_activated_classifier
    cl
    p1

    Returns
    ----------
    New behavioral classifier or None
    """
    if last_activated_classifier \
        and last_activated_classifier.does_anticipate_change() \
        and not last_activated_classifier.is_marked() \
        and cl.does_anticipate_change():
        nb_of_action = 1
        if last_activated_classifier.behavioral_sequence: 
            nb_of_action += len(last_activated_classifier.behavioral_sequence)
        if cl.behavioral_sequence: 
            nb_of_action += len(cl.behavioral_sequence)
        if  nb_of_action <= cl.cfg.bs_max:
            child = Classifier(
                action=last_activated_classifier.action, 
                behavioral_sequence=[],
                cfg=cl.cfg,
                quality=max(last_activated_classifier.q, 0.5))
            if last_activated_classifier.behavioral_sequence:
                child.behavioral_sequence.extend(last_activated_classifier.behavioral_sequence)
            child.behavioral_sequence.append(cl.action)
            if cl.behavioral_sequence:
                child.behavioral_sequence.extend(cl.behavioral_sequence)
            # Passthrough operation on child condition was not used because it can create not relevant classifiers. We prefer setting up the child condition the same as the last activated classifier.
            # Thus, we garantee the creation of a classifier that can be used within the environment.
            child.condition = last_activated_classifier.condition
            # Passthrough operation on child effect
            _updated_passthrough(child.effect, last_activated_classifier.effect, cl.effect, p1, child.condition)
            return child
    return None