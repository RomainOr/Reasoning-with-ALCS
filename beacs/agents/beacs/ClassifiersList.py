"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

import random
from itertools import chain
from typing import Optional, List

import beacs.agents.beacs.components.alp as alp
import beacs.agents.beacs.components.genetic_algorithms as ga
import beacs.agents.beacs.components.reinforcement_learning as rl
import beacs.agents.beacs.components.aliasing_detection as pai
from beacs import Perception, TypedList
from beacs.agents.beacs import Classifier, Configuration
from beacs.agents.beacs.components.add_classifier import add_classifier
from beacs.agents.beacs.components.build_behavioral_sequences import create_behavioral_classifier

class ClassifiersList(TypedList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args) -> None:
        super().__init__((Classifier, ), *args)


    def form_match_set(
            self,
            situation: Perception
        ) -> ClassifiersList:
        """
        Builds the ClassifiersList from the whole population with all classifiers whose condition
        matches the current situation.

        Parameters
        ----------
        situation: Perception
            Current perception

        Returns
        ----------
        ClassifiersList
            The whole set of matching classifiers
        """
        best_classifier = None
        best_fitness = 0.
        matching = []
        max_fitness_ra = 0.
        max_fitness_rb = 0.
        for cl in self:
            if cl.condition.does_match(situation):
                matching.append(cl)
                if cl.does_anticipate_change():
                    if cl.q*cl.ra > max_fitness_ra:
                        max_fitness_ra = cl.q*cl.ra
                    if cl.q*cl.rb > max_fitness_rb:
                        max_fitness_rb = cl.q*cl.rb
                    if cl.fitness > best_fitness:
                        best_classifier = cl
                        best_fitness = cl.fitness
        return ClassifiersList(*matching), best_classifier, max_fitness_ra, max_fitness_rb


    def form_action_set(
            self,
            action_classifier: Classifier
        ) -> ClassifiersList:
        """
        Builds the ClassifiersList from the match set with all classifiers whose actions
        match the ones of the selected classifier.

        Parameters
        ----------
        action_classifier: Classifier
            Classifier choosen by policies

        Returns
        ----------
        ClassifiersList
            The action set
        """
        matching = [cl for cl in self if cl.behavioral_sequence == action_classifier.behavioral_sequence and cl.action == action_classifier.action]
        return ClassifiersList(*matching)


    def expand(self) -> List[Classifier]:
        """
        Returns an array containing all micro-classifiers.

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
            time: int,
            cfg: Configuration
        ) -> None:
        candidates = [cl for cl in action_set if cl.ee]
        if len(candidates) < 2:
            return
        for i, cl1 in enumerate(candidates):
            for cl2 in candidates[i:]:
                if cl1.mark == cl2.mark and not cl1.effect.subsumes(cl2.effect) and not cl2.effect.subsumes(cl1.effect):
                    new_classifier = cl1.merge_with(cl2, time)
                    add_classifier(new_classifier, action_set, new_list)


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
        ) -> None:
        # First, try to detect if it is time to detect a pai state - no need to compute this every time
        match_set_no_bseq = [cl for cl in previous_match_set if cl.behavioral_sequence is None and (not cl.is_marked() or cl.mark.corresponds_to(p0))]
        if pai.should_pai_detection_apply(match_set_no_bseq, time, cfg.theta_bseq):
            # We set the related timestamp t_bseq of the classifiers in the match set
            pai.set_pai_detection_timestamps(match_set_no_bseq, time)
            # We check we have enough information from classifiers in the matching set to do the detection
            enough_information, most_experienced_classifiers = pai.enough_information_to_try_PAI_detection(match_set_no_bseq, cfg)
            if enough_information:
            # The system tries to determine is it suffers from the perceptual aliasing issue
                if pai.is_perceptual_aliasing_state(most_experienced_classifiers, p0, cfg) > 0:
                    # Add if needed the new pai state in memory
                    if p0 not in pai_states_memory:
                        #print("Add", p0, "in", pai_states_memory,'\n')
                        pai_states_memory.append(p0)
                else:
                    # Remove if needed the pai state from memory and delete all behavioral classifiers created for this state
                    if p0 in pai_states_memory:
                        #print("Remove", p0, "in", pai_states_memory,'\n')
                        pai_states_memory.remove(p0)
                        behavioral_classifiers_to_delete = [cl for cl in population if cl.pai_state == p0]
                        for cl in behavioral_classifiers_to_delete:
                            lists = [x for x in [population, match_set, action_set] if x]
                            for lst in lists:
                                lst.safe_remove(cl)

        # Create new behavioral classifiers
        if p0 in pai_states_memory and len(potential_cls_for_pai) > 0:
            pop_for_addition = [cl for cl in population if cl.behavioral_sequence and cl.condition.does_match(penultimate_classifier.condition)]
            for candidate in potential_cls_for_pai:
                new_cl = create_behavioral_classifier(penultimate_classifier, candidate, p1, p0, time)
                if new_cl:
                    add_classifier(new_cl, pop_for_addition, new_list)


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
                add_classifier(new_cl, action_set, new_list)

        # No classifier anticipated correctly - generate new one through covering
        # only if we are not in the case of behavioral sequences
        if not was_expected_case:
            if (len(action_set) > 0 and action_set[0].behavioral_sequence is None) or len(action_set) == 0:
                new_cl = alp.cover(p0, action, p1, time, cfg)
                add_classifier(new_cl, action_set, new_list)

        if cfg.do_pep:
            ClassifiersList.apply_enhanced_effect_part_check(action_set, new_list, time, cfg)

        if cfg.bs_max > 0 and penultimate_classifier is not None and len(potential_cls_for_pai) > 0:
            ClassifiersList.apply_perceptual_aliasing_issue_management(population, previous_match_set, match_set, action_set, penultimate_classifier, potential_cls_for_pai, new_list, p0, p1, time, pai_states_memory, cfg)

        # Merge classifiers from new_list into self and population
        population.extend(new_list)
        if match_set is not None:
            new_matching = [cl for cl in new_list if cl.condition.does_match(p1)]
            match_set.extend(new_matching)
        if action_set:
            new_action_cls = [cl for cl in new_list if cl.action == action_set[0].action and cl.behavioral_sequence == action_set[0].behavioral_sequence]
            action_set.extend(new_action_cls)


    @staticmethod
    def apply_reinforcement_learning(
            action_set: ClassifiersList,
            reward: int,
            max_fitness_ra: float,
            max_fitness_rb: float,
            beta_rl: float,
            gamma: float,
        ) -> None:
        for cl in action_set:
            #rl.update_classifier_q_learning(cl, reward, max_fitness_ra, beta_rl, gamma)
            rl.update_classifier_double_q_learning(cl, reward, max_fitness_ra, max_fitness_rb, beta_rl, gamma)


    @staticmethod
    def apply_ga(
            time: int,
            population: ClassifiersList,
            match_set: ClassifiersList,
            action_set: ClassifiersList,
            p0: Perception,
            p1: Perception,
            theta_ga: int,
            mu: float,
            chi: float,
            theta_as: int
        ) -> None:

        if ga.should_apply(action_set, time, theta_ga):
            ga.set_timestamps(action_set, time)

            # Select parents
            parent1, parent2 = ga.roulette_wheel_selection(
                action_set, 
                lambda cl: pow(cl.q, 3)
            )

            child1 = Classifier.copy_from(parent1, time)
            child2 = Classifier.copy_from(parent2, time)
            
            # Execute mutation
            ga.mutation(child1, child2, mu)

            # Execute cross-over
            if random.random() < chi:
                if child1.effect == child2.effect:
                    ga.two_point_crossover(child1, child2)

                    # Update quality and reward
                    child1.q = child2.q = (child1.q + child2.q) / 2.0
                    child1.ra = child2.ra = (child1.ra + child2.ra) / 2.0
                    child1.rb = child2.rb = (child1.rb + child2.rb) / 2.0

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
                    p1,
                    population,
                    match_set,
                    action_set
                )


    def __str__(self):
        return "\n".join(str(classifier)
            for classifier
            in sorted(self, key=lambda cl: -cl.fitness))
