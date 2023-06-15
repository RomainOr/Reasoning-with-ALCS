"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from typing import Optional
from agents.common.RandomNumberGenerator import RandomNumberGenerator


import agents.common.mechanisms.alp as alp_common
import agents.common.mechanisms.genetic_algorithms as ga
from agents.common.Perception import Perception
from agents.common.BaseClassifiersList import BaseClassifiersList
from agents.common.mechanisms.reinforcement_learning import update_classifier_q_learning

import agents.pepacs.mechanisms.alp as alp_pepacs
from agents.pepacs.PEPACSConfiguration import PEPACSConfiguration
from agents.pepacs.classifier_components.PEPACSClassifier import PEPACSClassifier


class PEPACSClassifiersList(BaseClassifiersList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args) -> None:
        super().__init__((PEPACSClassifier, ), *args)


    def form_match_set(
            self,
            situation: Perception
        ) -> PEPACSClassifiersList:
        """
        Builds the PEPACSClassifiersList from the whole population with all classifiers whose condition
        matches the current situation.

        Parameters
        ----------
        situation: Perception
            Current perception

        Returns
        ----------
        PEPACSClassifiersList
            The whole set of matching classifiers
        """
        best_classifier = None
        best_fitness = 0.0
        matching = []
        for cl in self:
            if cl.does_match(situation):
                matching.append(cl)
                if cl.does_anticipate_change() and cl.fitness > best_fitness:
                    best_classifier = cl
                    best_fitness = cl.fitness
        return PEPACSClassifiersList(*matching), best_classifier, best_fitness


    def form_action_set(
            self,
            action_classifier: PEPACSClassifier
        ):
        """
        Builds the ACS2ClassifiersList from the match set with all classifiers whose actions
        match the ones of the selected classifier.

        Parameters
        ----------
        action_classifier: Classifier
            Classifier choosen by policies

        Returns
        ----------
        ACS2ClassifiersList
            The action set
        """
        matching = [cl for cl in self if cl.behavioral_sequence == action_classifier.behavioral_sequence and cl.action == action_classifier.action]
        return PEPACSClassifiersList(*matching)


    @staticmethod
    def apply_alp(
            population: PEPACSClassifiersList,
            match_set: PEPACSClassifiersList,
            action_set: PEPACSClassifiersList,
            p0: Perception,
            action: int,
            p1: Perception,
            time: int,
            cfg: PEPACSConfiguration
        ) -> None:
        """
        The Anticipatory Learning Process. Handles all updates by the ALP,
        insertion of new classifiers in pop and possibly matchSet, and
        deletion of inadequate classifiers in pop and possibly matchSet.

        Parameters
        ----------
        population
        match_set
        action_set
        p0: Perception
        action: int
        p1: Perception
        time: int
        cfg: Configuration
        """
        new_list = PEPACSClassifiersList()
        new_cl: Optional[PEPACSClassifier] = None
        was_expected_case = False

        idx = 0
        action_set_length = 0
        if action_set: action_set_length = len(action_set)

        while(idx < action_set_length):
            cl = action_set[idx]
            cl.increase_experience()
            cl.set_alp_timestamp(time)

            if cl.does_anticipate_correctly(p0, p1):
                new_cl = alp_pepacs.expected_case(cl, p0, p1, time)
                was_expected_case = True
            else:
                new_cl = alp_common.unexpected_case(cl, p0, p1, time)

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
                new_cl.tga = time
                alp_common.add_classifier(new_cl, action_set, new_list)

        PEPACSClassifiersList.apply_enhanced_effect_part_check(action_set, new_list, p0, time, cfg)

        # No classifier anticipated correctly - generate new one
        if not was_expected_case:
            new_cl = alp_common.cover(PEPACSClassifier, p0, action, p1, time, cfg)
            alp_common.add_classifier(new_cl, action_set, new_list)

        # Merge classifiers from new_list into self and population
        PEPACSClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)


    @staticmethod
    def apply_reinforcement_learning(
            action_set: PEPACSClassifiersList,
            reward: int,
            max_fitness: float,
            cfg: PEPACSConfiguration
        ) -> None:
        for cl in action_set:
            update_classifier_q_learning(cl, reward, max_fitness, cfg.beta_rl, cfg.gamma)


    @staticmethod
    def apply_ga(
            population: PEPACSClassifiersList,
            match_set: PEPACSClassifiersList,
            action_set: PEPACSClassifiersList,
            p0: Perception,
            p1: Perception,
            time: int,
            cfg: PEPACSConfiguration
        ) -> None:
        ga.apply(
            PEPACSClassifiersList,
            PEPACSClassifier,
            ga.mutation,
            ga.two_point_crossover,
            population,
            match_set,
            action_set,
            p0,
            p1,
            time,
            cfg
        )
            

    @staticmethod
    def apply_enhanced_effect_part_check(
            action_set: PEPACSClassifiersList,
            new_list: PEPACSClassifiersList,
            previous_situation: Perception,
            time: int,
            cfg: PEPACSConfiguration
        ):
        # Create a list of candidates.
        # Every enhanceable classifier is a candidate.
        candidates = [cl for cl in action_set if cl.ee]
        # If there are less than 2 candidates, don't do it
        if len(candidates) < 2:
            return
        for candidate in candidates:
            candidates2 = [cl for cl in candidates if candidate != cl and cl.mark == candidate.mark]
            if len(candidates2) > 0:
                merger = RandomNumberGenerator.choice(candidates2)
                new_classifier = candidate.merge_with(merger, previous_situation, time)
                if new_classifier is not None:
                    alp_common.add_classifier(new_classifier, action_set, new_list)
        return new_list