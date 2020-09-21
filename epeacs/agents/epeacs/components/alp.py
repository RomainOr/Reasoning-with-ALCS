"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from random import random
from typing import Optional

from epeacs import Perception
from epeacs.agents.epeacs import Classifier, ClassifiersList, Condition, Configuration, PMark
from epeacs.agents.epeacs.ProbabilityEnhancedAttribute import ProbabilityEnhancedAttribute
from epeacs.agents.epeacs.components.aliasing_detection import is_state_aliased, is_perceptual_aliasing_state, set_pai_detection_timestamps, should_pai_detection_apply
from epeacs.agents.epeacs.components.build_behavioral_sequences import create_behavioral_classifier
from epeacs.agents.epeacs.components.subsumption import does_subsume


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
        penultimate_classifier: Classifier,
        cl: Classifier,
        p0: Perception,
        p1: Perception,
        time: int,
        previous_match_set: ClassifiersList,
        match_set: ClassifiersList,
        population: ClassifiersList,
        pai_states_memory,
        cfg: Configuration,
    ) -> Optional[Classifier]:
    """
    Controls the expected case of a classifier with the help of 
    Specification of Unchanging Components.

    Returns
    ----------
    New classifier or None
    """
    if cl.is_enhanced():
        cl.effect.update_enhanced_effect_probs(p1, cl.cfg.beta_pep)

    if is_state_aliased(cl.condition, cl.mark, p0):
        if cl.cfg.do_pep: cl.ee = True
        if cfg.bs_max > 0 and penultimate_classifier is not None:
            # Update the list of detetcted PAi states along with the population
            match_set_no_bseq = [cl for cl in previous_match_set if cl.behavioral_sequence is None]
            if should_pai_detection_apply(match_set_no_bseq, time, cfg.theta_bseq):
                set_pai_detection_timestamps(match_set_no_bseq, time)
                if is_perceptual_aliasing_state(match_set_no_bseq, p0, cfg):
                    if p0 not in pai_states_memory:
                        pai_states_memory.append(p0)
                    else:
                        behavioral_classifiers_to_zip = [cl for cl in population if cl.pai_state == p0]
                        behavioral_classifiers_to_delete = []
                        for idx1 in range(len(behavioral_classifiers_to_zip)-1):
                            cl = behavioral_classifiers_to_zip[idx1]
                            if cl in behavioral_classifiers_to_delete:
                                continue
                            for idx2 in range(idx1+1, len(behavioral_classifiers_to_zip)):
                                other_cl = behavioral_classifiers_to_zip[idx2]
                                if other_cl in behavioral_classifiers_to_delete:
                                    continue
                                if does_subsume(cl, other_cl, cfg.theta_exp) and cl.condition.subsumes(other_cl.condition):
                                    behavioral_classifiers_to_delete.append(other_cl)
                        for cl in behavioral_classifiers_to_delete:
                            population.safe_remove(cl)
                            match_set.safe_remove(cl)
                else:
                    if p0 in pai_states_memory:
                        pai_states_memory.remove(p0)
                        behavioral_classifiers_to_delete = [cl for cl in population if cl.pai_state == p0]
                        for cl in behavioral_classifiers_to_delete:
                            population.safe_remove(cl)
                            match_set.safe_remove(cl)
            # Create if needed a new behavioral classifier
            if p0 in pai_states_memory:
                child = create_behavioral_classifier(penultimate_classifier, cl, p1)
                if child:
                    child.tga = time
                    child.talp = time
                    child.pai_state = p0
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

    return child


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
