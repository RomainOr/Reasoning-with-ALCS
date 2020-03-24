"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from random import random
from typing import Optional

from bacs import Perception
from bacs.agents.bacs import Classifier, ClassifiersList, Condition, Configuration, PMark
from bacs.agents.bacs.ProbabilityEnhancedAttribute import ProbabilityEnhancedAttribute

def cover(
        p0: Perception,
        action: int,
        p1: Perception,
        time: int,
        cfg: Configuration
    ) -> Classifier:
    """
    Covering - creates a classifier that anticipates a change correctly.
    The reward of the new classifier is set to 0 to prevent *reward bubbles*
    in the environmental model.

    Parameters
    ----------
    p0: Perception
        previous perception
    action: int
        chosen action
    p1: Perception
        current perception
    time: int
        current epoch
    cfg: Configuration
        algorithm configuration class

    Returns
    -------
    Classifier
        new classifier
    """
    # In paper it's advised to set experience and reward of newly generated
    # classifier to 0. However in original code these values are initialized
    # with defaults 1 and 0.5 correspondingly.
    new_cl = Classifier(action=action, experience=1, reward=0.5, cfg=cfg)
    new_cl.tga = time
    new_cl.talp = time
    new_cl.specialize(p0, p1)
    return new_cl


def specification_of_unchanging_components_status(
        condition: Condition,
        mark: PMark,
        p0: Perception
    ) -> bool:
    """
    Check if the specification of unchanging components succeed or failed.
    Help to detect aliased states in POMDPs.

    Parameters
    ----------
    :param condition:
    :param mark:
    :param p0:

    Returns
    -------
    :return: bool
    """
    if mark.one_situation_in_mark():
        situation = Condition(condition)
        for idx, item in enumerate(condition):
            if item == condition.wildcard:
                situation[idx] = ''.join(str(s) for s in mark[idx])
        return not situation.does_match(p0)
    return True

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
    :param result: 
        The effect component to compute
    :param penultimate_classifier: 
    :param last_effect:
    :param perception:
    :param condition: 
        Condition component to remove unnecessary specification of effect attributes

    Returns
    -------
    :return: bool
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

    :param last_activated_classifier:
    :param cl:
    :param p1:
    :return: new behavioral classifier or None
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
            updated_passthrough(child.effect, last_activated_classifier.effect, cl.effect, p1, child.condition)
            return child
    return None

def expected_case(
        last_activated_classifier: Classifier,
        cl: Classifier,
        p0: Perception,
        p1: Perception,
        time: int
    ) -> Optional[Classifier]:
    """
    Controls the expected case of a classifier with the help of 
    Specification of Unchanging Components.
    Controls also the case when the specification of unchanging
    components failed by creating classifiers with action chunks,
    only if the action set is not a behavioral action set.

    :param last_activated_classifier:
    :param cl:
    :param p0:
    :param p1:
    :param time:
    :return: new classifier or None
    """
    if cl.ee:
        cl.effect.update_enhanced_effect_probs(p0, cl.cfg.beta)

    if not specification_of_unchanging_components_status(cl.condition, cl.mark, p0):
        if cl.cfg.do_pee: cl.ee = True
        if cl.cfg.bs_max != 0 and last_activated_classifier is not None:
            child = create_behavioral_classifier(last_activated_classifier, cl, p1)
            if child:
                child.tga = time
                child.talp = time
                return child

    diff = cl.mark.get_differences(p0)
    if diff.specificity == 0:
        cl.increase_quality()
        return None

    child = cl.copy_from(cl, p1, time)

    no_spec = len(cl.specified_unchanging_attributes)
    no_spec_new = diff.specificity
    if no_spec >= cl.cfg.u_max:
        while no_spec >= cl.cfg.u_max:
            res = cl.generalize_unchanging_condition_attribute()
            assert res is True
            no_spec -= 1

        while no_spec + no_spec_new > cl.cfg.u_max:
            if random() < 0.5:
                diff.generalize_specific_attribute_randomly()
                no_spec_new -= 1
            else:
                if cl.generalize_unchanging_condition_attribute():
                    no_spec -= 1
    else:
        while no_spec + no_spec_new > cl.cfg.u_max:
            diff.generalize_specific_attribute_randomly()
            no_spec_new -= 1

    child.condition.specialize_with_condition(diff)

    if child.q < 0.5:
        child.q = 0.5

    return child

def unexpected_case(
        cl: Classifier,
        p0: Perception,
        p1: Perception,
        time: int
    ) -> Optional[Classifier]:
    """
    Controls the unexpected case of the classifier.

    :param cl:
    :param p0:
    :param p1:
    :param time:
    :return: specialized classifier if generation was possible,
    None otherwise
    """
    cl.decrease_quality()
    cl.set_mark(p0)
    if not cl.effect.is_specializable(p0, p1):
        return None
    child = cl.copy_from(cl, p1, time)
    child.specialize(p0, p1)
    if child.q < 0.5:
        child.q = 0.5
    return child
