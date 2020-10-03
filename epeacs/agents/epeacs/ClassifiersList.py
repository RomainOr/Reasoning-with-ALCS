"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

import random
from itertools import chain
from typing import Optional, List

import epeacs.agents.epeacs.components.alp as alp
import epeacs.agents.epeacs.components.genetic_algorithms as ga
import epeacs.agents.epeacs.components.reinforcement_learning as rl
import epeacs.agents.epeacs.components.aliasing_detection as pai
from epeacs import Perception, TypedList
from epeacs.agents.epeacs import Classifier, Configuration
from epeacs.agents.epeacs.components.add_classifier import add_classifier
from epeacs.agents.epeacs.components.build_behavioral_sequences import create_behavioral_classifier

class ClassifiersList(TypedList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args) -> None:
        super().__init__((Classifier, ), *args)


    def form_match_set(self, situation: Perception) -> ClassifiersList:
        best_classifier = None
        best_fitness = 0.0
        matching = []
        for cl in self:
            if cl.condition.does_match(situation):
                matching.append(cl)
                # Based on hypothesis that a change should be anticipted
                if cl.does_anticipate_change() and cl.fitness > best_fitness:
                #if cl.fitness > best_fitness:
                    best_classifier = cl
                    best_fitness = cl.fitness
        return ClassifiersList(*matching), best_classifier


    def form_action_set(self, action_classifier: Classifier) -> ClassifiersList:
        matching = [cl for cl in self if cl.behavioral_sequence == action_classifier.behavioral_sequence and cl.action == action_classifier.action]
        return ClassifiersList(*matching)


    def expand(self) -> List[Classifier]:
        """
        Returns an array containing all micro-classifiers

        Returns
        -------
        List[Classifier]
            list of all expanded classifiers
        """
        list2d = [[cl] * cl.num for cl in self]
        return list(chain.from_iterable(list2d))


    @staticmethod
    def apply_enhanced_effect_part_check(
            action_set: ClassifiersList,
            new_list: ClassifiersList,
            previous_situation: Perception,
            time: int,
            cfg: Configuration
        ):
        candidates = [cl for cl in action_set if cl.ee]
        if len(candidates) < 2:
            return
        for candidate in candidates:
            candidates2 = [cl for cl in candidates if candidate != cl and cl.mark == candidate.mark]
            if len(candidates2) > 0:
                merger = random.choice(candidates2)
                new_classifier = candidate.merge_with(merger, previous_situation, time)
                add_classifier(new_classifier, action_set, new_list, cfg.theta_exp)


    @staticmethod
    def apply_perceptual_aliasing_issue_management(
            population: ClassifiersList,
            previous_match_set: ClassifiersList,
            match_set: ClassifiersList,
            action_set: ClassifiersList,
            penultimate_classifier: Classifier,
            potential_cls_for_pai: List(Classifier),
            new_list: ClassifiersList,
            p0: Perception,
            p1: Perception,
            time: int,
            pai_states_memory,
            cfg: Configuration
        ):
        # First, try to detect a PAI state if needed
        match_set_no_bseq = [cl for cl in previous_match_set if cl.behavioral_sequence is None]
        if pai.should_pai_detection_apply(match_set_no_bseq, time, cfg.theta_bseq):
            pai.set_pai_detection_timestamps(match_set_no_bseq, time)
            # The system tries to determine is it suffers from the perceptual aliasing issue
            if pai.is_perceptual_aliasing_state(match_set_no_bseq, p0, cfg):
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
        if p0 in pai_states_memory:
            for candidate in potential_cls_for_pai:
                new_cl = create_behavioral_classifier(penultimate_classifier, candidate, p1, p0, time)
                if new_cl:
                    pop_for_addition = [cl for cl in population if cl.behavioral_sequence and cl.condition.does_match(new_cl.condition)]
                    add_classifier(new_cl, pop_for_addition, new_list, cfg.theta_exp)


    @staticmethod
    def apply_alp(
            population: ClassifiersList,
            previous_match_set: ClassifiersList,
            match_set: ClassifiersList,
            action_set: ClassifiersList,
            penultimate_classifier: Classifier,
            p0: Perception,
            action: int,
            p1: Perception,
            time: int,
            pai_states_memory,
            cfg: Configuration
        ) -> None:
        """
        The Anticipatory Learning Process. Handles all updates by the ALP,
        insertion of new classifiers in pop and possibly matchSet, and
        deletion of inadequate classifiers in pop and possibly matchSet.

        Parameters
        ----------
        population
        previous_match_set
        match_set
        action_set
        penultimate_classifier
        p0: Perception
        action: int
        p1: Perception
        time: int
        pai_states_memory
        cfg: Configuration
        """
        new_list = ClassifiersList()
        new_cl: Optional[Classifier] = None
        was_expected_case = False

        idx = 0
        action_set_length = 0
        if action_set: action_set_length = len(action_set)
        if cfg.bs_max > 0 and penultimate_classifier is not None: potential_cls_for_pai = []

        #Main ALP loop on the action set
        while(idx < action_set_length):
            cl = action_set[idx]
            cl.increase_experience()
            cl.set_alp_timestamp(time)
            is_aliasing_detected = False

            if cl.does_anticipate_correctly(p0, p1):
                is_aliasing_detected, new_cl = alp.expected_case(cl, p0, p1, time, cfg)
                was_expected_case = True
                if cfg.bs_max > 0 and penultimate_classifier is not None and is_aliasing_detected:
                    potential_cls_for_pai.append(cl)
            else:
                new_cl = alp.unexpected_case(cl, p0, p1, time)

            if cl.is_inadequate():
                # Removes classifier from population, match set
                # and current list
                lists = [x for x in [population, match_set, action_set] if x]
                for lst in lists:
                    lst.safe_remove(cl)
                idx -= 1
                action_set_length -= 1
            idx += 1

            if new_cl is not None:
                add_classifier(new_cl, action_set, new_list, cfg.theta_exp)

        # No classifier anticipated correctly - generate new one through covering
        # only if we are not in the case of behavioral sequences
        if not was_expected_case:
            if (len(action_set) > 0 and action_set[0].behavioral_sequence is None) or len(action_set) == 0:
                new_cl = alp.cover(p0, action, p1, time, cfg)
                add_classifier(new_cl, action_set, new_list, cfg.theta_exp)

        if cfg.do_pep:
            ClassifiersList.apply_enhanced_effect_part_check(action_set, new_list, p0, time, cfg)

        if cfg.bs_max > 0 and penultimate_classifier is not None:
            ClassifiersList.apply_perceptual_aliasing_issue_management(population, previous_match_set, match_set, action_set, penultimate_classifier, potential_cls_for_pai, new_list, p0, p1, time, pai_states_memory, cfg)

        # Merge classifiers from new_list into self and population
        action_set.extend(new_list)
        population.extend(new_list)

        if match_set is not None:
            new_matching = [cl for cl in new_list if cl.condition.does_match(p1)]
            match_set.extend(new_matching)


    @staticmethod
    def apply_reinforcement_learning(
            match_set: ClassifiersList,
            action_set: ClassifiersList,
            reward: int,
            beta_rl: float,
            gamma: float,
            done: bool
        ) -> None:
        max_fitness_ra = 0.
        max_fitness_rb = 0.
        if not done:
            for cl in match_set:
                if cl.does_anticipate_change():
                    if cl.q*cl.ra > max_fitness_ra:
                        max_fitness_ra = cl.q*cl.ra
                    if cl.q*cl.rb > max_fitness_rb:
                        max_fitness_rb = cl.q*cl.rb
        for cl in action_set:
            #rl.update_classifier_q_learning(cl, reward, max_fitness_ra, beta_rl, gamma)
            rl.update_classifier_double_q_learning(cl, reward, max_fitness_ra, max_fitness_rb, beta_rl, gamma)


    @staticmethod
    def apply_ga(
            time: int,
            population: ClassifiersList,
            match_set: ClassifiersList,
            action_set: ClassifiersList,
            p: Perception,
            theta_ga: int,
            mu: float,
            chi: float,
            theta_as: int,
            theta_exp: int
        ) -> None:

        if ga.should_apply(action_set, time, theta_ga):
            ga.set_timestamps(action_set, time)

            # Select parents
            parent1, parent2 = ga.roulette_wheel_selection(
                action_set, 
                lambda cl: pow(cl.q, 3) * cl.num
            )

            child1 = Classifier.copy_from(parent1, p, time)
            child2 = Classifier.copy_from(parent2, p, time)

            # Execute mutation
            ga.generalizing_mutation(child1, mu)
            ga.generalizing_mutation(child2, mu)

            # Execute cross-over
            if random.random() < chi:
                if child1.effect == child2.effect:
                    ga.two_point_crossover(child1, child2)

                    # Update quality and reward
                    child1.q = child2.q = float(sum([child1.q, child2.q]) / 2)
                    child1.ra = child2.ra = float(sum([child1.ra, child2.ra]) / 2)
                    child1.rb = child2.rb = float(sum([child1.rb, child2.rb]) / 2)

            child1.q /= 2
            child2.q /= 2

            # We are interested only in classifiers with specialized condition
            unique_children = {cl for cl in [child1, child2]
                               if cl.condition.specificity > 0}

            ga.delete_classifiers(
                population,
                match_set,
                action_set,
                len(unique_children),
                theta_as
            )

            # check for subsumers / similar classifiers
            for child in unique_children:
                ga.add_classifier(
                    child,
                    p,
                    population,
                    match_set,
                    action_set,
                    theta_exp
                )


    def __str__(self):
        return "\n".join(str(classifier)
            for classifier
            in sorted(self, key=lambda cl: -cl.fitness))
