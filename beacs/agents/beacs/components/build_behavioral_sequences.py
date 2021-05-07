"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Optional

from beacs import Perception
from beacs.agents.beacs import Classifier

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

    Parameters
    ----------
    child_effect
        The effect component to compute and the result
    penultimate_effect
        The effect component of the penultimate classifier
    last_effect
        The effect component of the last classifier
    perception
        The current perception to refine the effect component of the child
    child_condition
        Condition component to remove unnecessary specification of effect attributes
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
        penultimate_classifier: Classifier,
        cl: Classifier,
        p1: Perception,
        pai_state: Perception,
        time: int
    ) -> Optional[Classifier]:
    """
    Builds a behavioral classifier.

    Parameters
    ----------
    penultimate_classifier: Classifier
        Penultimate classifier selected through classifier selection previously
    cl: Classifier
        Current classifier of the action set
    p1: Perception
        Perception received after the action is done
    pai_state: Perception
        Perception of the Perceptual Aliasing State
    time: int
        Curretn epoch

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
                rewarda=cl.ra,
                rewardb=cl.rb,
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
