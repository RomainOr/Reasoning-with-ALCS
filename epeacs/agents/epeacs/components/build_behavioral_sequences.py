"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Optional

from epeacs import Perception
from epeacs.agents.epeacs import Classifier

def updated_passthrough(
        child_effect, 
        penultimate_effect, 
        last_effect, 
        perception,
        child_condition
    ):
    """
    Passthrough operator defined by Stolzmann that we have refined.
    It is only used on the effect component of classifiers.
    The target is to get first a behavioral classifier that can bridge the aliased state.
    As a consequence, we do not enable the building of a behavioral enhanced classifier.

    Parameters
    ----------
    child_effect
        The effect component to compute and the result
    penultimate_effect
    last_effect
    perception
    child_condition
        Condition component to remove unnecessary specification of effect attributes
    """
    for i in range(len(child_effect)):
        if last_effect.is_enhanced():
            child_effect[i] = perception[i]
        elif last_effect[i] == child_effect.wildcard:
            if penultimate_effect.is_enhanced():
                child_effect[i] = perception[i]
            else:
                child_effect[i] = penultimate_effect[i]
        else:
            child_effect[i] = last_effect[i]
    # Refining effect
    for idx, effect_item in enumerate(child_effect):
        if effect_item != child_effect.wildcard and effect_item == child_condition[idx]:
            child_effect[idx] = child_effect.wildcard

def create_behavioral_classifier(
        penultimate_classifier: Classifier,
        cl: Classifier,
        p1: Perception
    ) -> Optional[Classifier]:
    """
    Build a behavioral classifier.

    Parameters
    ----------
    penultimate_classifier
    cl
    p1

    Returns
    ----------
    New behavioral classifier or None
    """
    if penultimate_classifier \
        and penultimate_classifier.does_anticipate_change() \
        and not penultimate_classifier.is_marked() \
        and cl.does_anticipate_change():
        nb_of_action = 1
        if penultimate_classifier.behavioral_sequence: 
            nb_of_action += len(penultimate_classifier.behavioral_sequence)
        if cl.behavioral_sequence: 
            nb_of_action += len(cl.behavioral_sequence)
        if nb_of_action <= cl.cfg.bs_max:
            child = Classifier(
                action=penultimate_classifier.action, 
                behavioral_sequence=[],
                cfg=cl.cfg,
                quality=max(cl.q, penultimate_classifier.q),
                reward=cl.r
            )
            if penultimate_classifier.behavioral_sequence:
                child.behavioral_sequence.extend(penultimate_classifier.behavioral_sequence)
            child.behavioral_sequence.append(cl.action)
            if cl.behavioral_sequence:
                child.behavioral_sequence.extend(cl.behavioral_sequence)
            # Passthrough operation on child condition was not used because it can create not relevant classifiers. We prefer setting up the child condition the same as the penultimate activated classifier.
            # Thus, we garantee the creation of a classifier that can be used within the environment.
            child.condition = penultimate_classifier.condition
            # Passthrough operation on child effect
            updated_passthrough(child.effect, penultimate_classifier.effect, cl.effect, p1, child.condition)
            return child
    return None
