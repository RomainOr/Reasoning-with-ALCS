"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Optional

from agents.common.Perception import Perception
from agents.common.classifier_components.Condition import Condition
from agents.common.classifier_components.Effect import Effect

from agents.beacs.classifier_components.EffectList import EffectList
from agents.beacs.classifier_components.BEACSClassifier import BEACSClassifier


def updated_passthrough(
        child_effect: Effect,
        penultimate_effect: EffectList, 
        last_effect: EffectList, 
        perception: Perception,
        child_condition: Condition
    ) -> None:
    """
    Passthrough operator inspired by Stolzmann's one.
    It is only used on the effect component of classifiers.

    Parameters
    ----------
        child_effect: Effect
        penultimate_effect: EffectList
        last_effect: EffectList
        perception: Perception
        child_condition: Condition
    """
    for i in range(len(child_effect)):
        change_anticipated = last_effect[0][i] != child_effect.wildcard
        if last_effect.is_enhanced():
            for effect in last_effect:
                if effect[i] != effect.wildcard and effect[i] == perception[i]:
                    change_anticipated = True
                    break
        if change_anticipated :
            child_effect[i] = perception[i]
        else :
            change_anticipated = penultimate_effect[0][i] != child_effect.wildcard
            if penultimate_effect.is_enhanced():
                for effect in penultimate_effect:
                    if effect[i] != effect.wildcard and effect[i] == perception[i]:
                        change_anticipated = True
                        break
            if change_anticipated :
                child_effect[i] = perception[i]
    # Refining effect
    for idx in range(len(child_effect)):
        if child_effect[idx] != child_effect.wildcard and child_effect[idx] == child_condition[idx]:
            child_effect[idx] = child_effect.wildcard


def create_behavioral_classifier(
        penultimate_classifier: BEACSClassifier,
        cl: BEACSClassifier,
        p1: Perception,
        pai_state: Perception,
        time: int
    ) -> Optional[BEACSClassifier]:
    """
    Builds a behavioral classifier.

    Parameters
    ----------
        penultimate_classifier: BEACSClassifier
        cl: BEACSClassifier
        p1: Perception
        pai_state: Perception
        time: int

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
            child = BEACSClassifier(
                action=penultimate_classifier.action, 
                behavioral_sequence=[],
                reward=cl.r,
                reward_bis=cl.r_bis,
                tga=time,
                tbseq=time,
                talp=time,
                pai_state=pai_state,
                cfg=cl.cfg
            )
            if penultimate_classifier.behavioral_sequence:
                child.behavioral_sequence.extend(penultimate_classifier.behavioral_sequence)
            child.behavioral_sequence.append(cl.action)
            if cl.behavioral_sequence:
                child.behavioral_sequence.extend(cl.behavioral_sequence)
            # Passthrough operation on child condition was not used because it can create not relevant classifiers. We prefer setting up the child condition the same as the penultimate activated classifier.
            # Thus, we garantee the creation of a classifier that can be used within the environment.
            child.condition.specialize_with_condition(penultimate_classifier.condition)
            # Passthrough operation on child effect
            updated_passthrough(child.effect[0], penultimate_classifier.effect, cl.effect, p1, child.condition)
            return child
    return None
