"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from random import random
from typing import Optional

from epeacs import Perception
from epeacs.agents.epeacs import Classifier, ClassifiersList, Condition, Configuration, PMark
from epeacs.agents.epeacs.components.aliasing_detection import is_state_aliased


def cover(
        p0: Perception,
        action: int,
        p1: Perception,
        time: int,
        cfg: Configuration
    ) -> Classifier:
    """
    Covering - creates a classifier that anticipates a change correctly.

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
    new_cl = Classifier(
        action=action, 
        tga=time,
        tbseq=time,
        talp=time,
        cfg=cfg
    )
    new_cl.specialize(p0, p1)
    return new_cl


def expected_case(
        cl: Classifier,
        p0: Perception,
        p1: Perception,
        time: int,
        cfg: Configuration,
    ):
    """
    Controls the expected case of a classifier with the help of 
    Specification of Unchanging Components.

    Returns
    ----------
    New classifier or None
    """
    is_aliasing_detected = False

    if cl.is_enhanced():
        cl.effect.update_effect_detailled_counter(p0, p1)
        is_aliasing_detected = True

    if is_state_aliased(cl.condition, cl.mark, p0):
        if cl.cfg.do_pep: cl.ee = True
        is_aliasing_detected = True

    diff = cl.mark.get_differences(p0)
    if diff.specificity == 0:
        cl.increase_quality()
        return is_aliasing_detected, None

    child = cl.copy_from(cl, p1, time)

    no_spec = len(cl.specified_unchanging_attributes)
    no_spec_new = diff.specificity
    if no_spec >= cl.cfg.u_max:
        while no_spec >= cl.cfg.u_max:
            cl.generalize_unchanging_condition_attribute()
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

    return is_aliasing_detected, child


def unexpected_case(
        cl: Classifier,
        p0: Perception,
        p1: Perception,
        time: int
    ) -> Optional[Classifier]:
    """
    Controls the unexpected case of the classifier.

    Returns
    ----------
    Specialized classifier if generation was possible, otherwise None
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
