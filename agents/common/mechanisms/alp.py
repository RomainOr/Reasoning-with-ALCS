"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Optional
if TYPE_CHECKING:
    from agents.common.BaseClassifiersList import BaseClassifiersList

from agents.common.BaseConfiguration import BaseConfiguration
from agents.common.Perception import Perception
from agents.common.RandomNumberGenerator import RandomNumberGenerator
from agents.common.classifier_components.Condition import Condition
from agents.common.classifier_components.BaseClassifier import BaseClassifier
from agents.common.mechanisms.subsumption import does_subsume


def cover(
        cls: BaseClassifier,
        p0: Perception,
        action: int,
        p1: Perception,
        time: int,
        cfg: BaseConfiguration
    ) -> BaseClassifier:
    """
    Covering - creates a classifier that anticipates a change correctly.

    Parameters
    ----------
        cls: BaseClassifier
        p0: Perception
        action: int
        p1: Perception
        time: int
        cfg: BaseConfiguration

    Returns
    -------
    BaseClassifier
    """
    new_cl = cls(
        action=action, 
        tga=time,
        talp=time,
        cfg=cfg
    )
    new_cl.specialize(p0, p1)
    return new_cl


def expected_case(
        cl: BaseClassifier,
        p0: Perception,
        time: int,
        p1: Perception=None
    ) -> Optional[BaseClassifier]:
    """
    Controls the expected case of a classifier with the help of 
    Specification of Unchanging Components.

    Parameters
    ----------
        cls: BaseClassifier
        p0: Perception
        time: int
        p1: Perception

    Returns
    ----------
    BaseClassifier
    """
    diff = cl.mark.get_differences(p0)
    if diff.specificity == 0:
        cl.increase_quality()
        return None

    return specification_unchanging_components(cl, diff, time, p1)


def specification_unchanging_components(
        cl: BaseClassifier,
        diff: Condition,
        time: int,
        p1: Perception=None
    ) -> BaseClassifier:
    """
    Controls the expected case of a classifier with the help of 
    Specification of Unchanging Components.

    Parameters
    ----------
        cls: BaseClassifier
        p0: Perception
        time: int
        p1: Perception

    Returns
    ----------
    BaseClassifier
    """
    child = cl.copy(time=time, perception=p1)
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
        cl: BaseClassifier,
        p0: Perception,
        p1: Perception,
        time: int
    ) -> Optional[BaseClassifier]:
    """
    Controls the unexpected case of the classifier.

    Parameters
    ----------
        cls: BaseClassifier
        p0: Perception
        p1: Perception
        time: int

    Returns
    ----------
    BaseClassifier
    """
    cl.decrease_quality()
    cl.set_mark(p0)
    if not cl.is_specializable(p0, p1):
        return None
    child = cl.copy(time=time, perception=p1)
    child.specialize(p0, p1)
    child.q = max(0.5, child.q)
    return child


def add_classifier(
        child: BaseClassifier, 
        population: BaseClassifiersList,
        new_list: list(BaseClassifier)
    ) -> None:
    """
    Looks for subsuming / similar classifiers in the population of classifiers
    and those created in the current ALP run (`new_list`).

    If a similar classifier was found it's quality is increased,
    otherwise `child_cl` is added to `new_list`.

    Parameters
    ----------
        child: BaseClassifier
        population: BaseClassifiersList
        new_list: list(BaseClassifier)
    """
    old_cl = None
    equal_cl = None
    # Look if there is a classifier that subsumes the insertion candidate
    for cl in population:
        if does_subsume(cl, child):
            if old_cl is None or cl.is_more_general(old_cl):
                old_cl = cl
        elif cl == child:
            equal_cl = cl
    # Check if there is similar classifier already in the population, previously found
    if old_cl is None:
        old_cl = equal_cl
    # Check if any similar classifier was in this ALP run
    if old_cl is None:
        for cl in new_list:
            if cl == child:
                old_cl = cl
                break
    if old_cl is None:
        new_list.append(child)
    else:
        old_cl.increase_quality()