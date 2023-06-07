"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from operator import attrgetter
from typing import Optional
from agents.common.classifier_components.BaseClassifier import BaseClassifier

import agents.common.mechanisms.genetic_algorithms as ga
from agents.common.Perception import Perception
from agents.common.RandomNumberGenerator import RandomNumberGenerator
from agents.common.BaseClassifiersList import BaseClassifiersList

import agents.beacs.mechanisms.alp as alp
import agents.beacs.mechanisms.reinforcement_learning as rl
from agents.beacs.BEACSConfiguration import BEACSConfiguration
from agents.beacs.classifier_components.BEACSClassifier import BEACSClassifier


class BEACSClassifiersList(BaseClassifiersList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args) -> None:
        super().__init__((BEACSClassifier, ),*args)


    def form_match_set(
            self,
            situation: Perception
        ) -> BEACSClassifiersList:
        """
        Builds the BEACSClassifiersList from the whole population with all classifiers whose condition
        matches the current situation.

        Parameters
        ----------
        situation: Perception
            Current perception

        Returns
        ----------
        BEACSClassifiersList
            The whole set of matching classifiers
        """
        matching = [cl for cl in self if cl.does_match(situation)]
        matching_with_change_anticipated = [cl for cl in matching if cl.does_anticipate_change()]
        best_classifier = max(matching_with_change_anticipated,key=attrgetter('fitness'),default=None)
        max_fitness_r = max((cl.q*cl.r for cl in matching_with_change_anticipated), default=0.)
        max_fitness_r_bis = max((cl.q*cl.r_bis for cl in matching_with_change_anticipated), default=0.)
        return BEACSClassifiersList(*matching), best_classifier, max_fitness_r, max_fitness_r_bis
    

    def form_action_set(
            self,
            action_classifier: BaseClassifier
        ):
        """
        Builds theBaseClassifiersList from the match set with all classifiers whose actions
        match the ones of the selected classifier.

        Parameters
        ----------
        action_classifier: Classifier
            Classifier choosen by policies

        Returns
        ----------
        The action set
        """
        matching = [cl for cl in self if cl.behavioral_sequence == action_classifier.behavioral_sequence and cl.action == action_classifier.action]
        return BEACSClassifiersList(*matching)


    @staticmethod
    def apply_alp(
            population: BEACSClassifiersList,
            t_2_match_set: BEACSClassifiersList,
            t_1_match_set: BEACSClassifiersList,
            match_set: BEACSClassifiersList,
            action_set: BEACSClassifiersList,
            penultimate_classifier: BEACSClassifier,
            p0: Perception,
            action: int,
            p1: Perception,
            time: int,
            pai_states_memory,
            cfg: BEACSConfiguration
        ) -> None:
        """
        The Anticipatory Learning Process. Handles all updates by the ALP,
        insertion of new classifiers in pop and possibly matchSet, and
        deletion of inadequate classifiers in pop and possibly matchSet.

        Parameters
        ----------
        population
        t_2_match_set
        t_1_match_set
        match_set
        action_set
        penultimate_classifier
        p0: Perception
        action: int
        p1: Perception
        time: int
        pai_states_memory
        cfg:BEACSConfiguration
        """
        new_list = BEACSClassifiersList()
        new_cl: Optional[BEACSClassifier] = None
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
                if new_cl.does_match(p0):
                    alp.add_classifier(new_cl, action_set, new_list)
                else:
                    alp.add_classifier(new_cl, population, new_list)

        # No classifier anticipated correctly - generate new one through covering
        # only if we are not in the case of classifiers having behavioral sequences
        if not was_expected_case:
            if (len(action_set) > 0 and action_set[0].behavioral_sequence is None) or len(action_set) == 0:
                new_cl = alp.cover(p0, action, p1, time, cfg)
                alp.add_classifier(new_cl, action_set, new_list)

        if cfg.do_ep:
            alp.apply_enhanced_effect_part_check(action_set, new_list, p0, time)

        if cfg.bs_max > 0 and penultimate_classifier is not None and len(potential_cls_for_pai) > 0:
            alp.apply_perceptual_aliasing_issue_management(population, t_2_match_set, t_1_match_set, match_set, action_set, penultimate_classifier, potential_cls_for_pai, new_list, p0, p1, time, pai_states_memory, cfg)

        # Merge classifiers from new_list into self and population
        BEACSClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)


    @staticmethod
    def apply_reinforcement_learning(
            action_set: BEACSClassifiersList,
            reward: int,
            max_fitness_ra: float,
            max_fitness_rb: float,
            beta_rl: float,
            gamma: float
        ) -> None:
        for cl in action_set:
            rl.update_classifier_double_q_learning(cl, reward, max_fitness_ra, max_fitness_rb, beta_rl, gamma)


    @staticmethod
    def apply_ga(
            time: int,
            population: BEACSClassifiersList,
            match_set: BEACSClassifiersList,
            action_set: BEACSClassifiersList,
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

            child1 = BEACSClassifier.copy_from(parent1, time)
            child2 = BEACSClassifier.copy_from(parent2, time)
            
            # Execute mutation
            ga.mutation(child1, child2, mu)

            # Execute cross-over
            if RandomNumberGenerator.random() < chi:
                if child1.effect == child2.effect:
                    ga.two_point_crossover(child1, child2)

                    # Update quality and reward
                    child1.q = child2.q = (child1.q + child2.q) / 2.0
                    child1.r = child2.r = (child1.r + child2.r) / 2.0
                    child1.r_bis = child2.r_bis = (child1.r_bis + child2.r_bis) / 2.0

            child1.q /= 2
            child2.q /= 2

            # We are interested only in classifiers with specialized condition
            children = {cl for cl in [child1, child2]
                               if cl.condition.specificity > 0}

            ga.delete_classifiers(
                population,
                match_set,
                action_set,
                len(children),
                theta_as
            )

            new_list = BEACSClassifiersList()
            # check for subsumers / similar classifiers
            for child in children:
                ga.add_classifier(
                    child,
                    action_set,
                    new_list
                )
            # Merge classifiers from new_list into self and population
            BEACSClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)