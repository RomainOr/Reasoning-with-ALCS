"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Optional

from pepacs import Perception, RandomNumberGenerator
from pepacs.agents.pepacs import Configuration
from pepacs.agents.pepacs.classifier_components import Classifier
from pepacs.agents.pepacs.components.aliasing_detection import is_state_aliased
from pepacs.agents.pepacs.components.subsumption import does_subsume


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


def expected_case(
        cl: Classifier,
        p0: Perception,
        p1: Perception,
        time: int
    ) -> Optional[Classifier]:
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

    if is_state_aliased(cl.condition, cl.mark, p0):
        if cl.cfg.do_pep: cl.ee = True

    diff = cl.mark.get_differences(p0)
    if diff.specificity == 0:
        cl.increase_quality()
        return None

    child = cl.copy_from(cl, p1, time, False)

    spec = cl.specificity
    spec_new = diff.specificity
    if spec >= child.cfg.u_max:
        while spec >= child.cfg.u_max:
            child.generalize_specific_attribute_randomly()
            spec -= 1
        while spec + spec_new > child.cfg.u_max:
            if spec > 0 and RandomNumberGenerator.random() < 0.5:
                child.generalize_specific_attribute_randomly()
                spec -= 1
            else:
                diff.generalize_specific_attribute_randomly()
                spec_new -= 1
    else:
        while spec + spec_new > child.cfg.u_max:
            diff.generalize_specific_attribute_randomly()
            spec_new -= 1


    child.condition.specialize_with_condition(diff)

    child.q = max(0.5, child.q)

    return child


def unexpected_case(
        cl: Classifier,
        p0: Perception,
        p1: Perception,
        time: int
    ) -> Optional[Classifier]:
    """
    Controls the unexpected case of the classifier.

    Parameters
    ----------
    cl
    p0
    p1
    time

    Returns
    ----------
    Specialized classifier if generation was possible, otherwise None
    """
    cl.decrease_quality()
    cl.set_mark(p0)
    if not cl.effect.is_specializable(p0, p1):
        return None
    child = cl.copy_from(cl, p1, time, False)
    child.specialize(p0, p1)
    if child.q < 0.5:
        child.q = 0.5
    return child


def add_classifier(child, population, new_list, theta_exp: int) -> None:
    """
    Looks for subsuming / similar classifiers in the population of classifiers
    and those created in the current ALP run (`new_list`).

    If a similar classifier was found it's quality is increased,
    otherwise `child_cl` is added to `new_list`.

    Parameters
    ----------
    child:
        New classifier to examine
    population:
        list of classifiers
    new_list:
        A list of newly created classifiers in this ALP run
    theta_exp: int
        experience threshold for subsumption
    """
    old_cl = None

    # Look if there is a classifier that subsumes the insertion candidate
    for cl in population:
        if does_subsume(cl, child, theta_exp):
            if old_cl is None or cl.is_more_general(old_cl):
                old_cl = cl
                break

    # Check if any similar classifier was in this ALP run
    if old_cl is None:
        for cl in new_list:
            if cl == child:
                old_cl = cl
                break

    # Check if there is similar classifier already
    if old_cl is None:
        for cl in population:
            if cl == child:
                old_cl = cl
                break

    if old_cl is None:
        new_list.append(child)
    else:
        old_cl.increase_quality()