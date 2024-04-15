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
from agents.common.classifier_components.BaseClassifier import BaseClassifier
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
        The Anticipatory Learning Process. Handles insertion, update and deletion
        of new classifiers in population and possibly other sets.

        Parameters
        ----------
            population: PEPACSClassifiersList
            match_set: PEPACSClassifiersList
            action_set: PEPACSClassifiersList
            p0: Perception
            action: int
            p1: Perception
            time: int
            cfg: PEPACSConfiguration
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

        alp_pepacs.apply_enhanced_effect_part_check(action_set, new_list, p0, time)

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
        """
        The Reinforcement Learning Process. Handles all reward updates.

        Parameters
        ----------
            action_set: PEPACSClassifiersList
            reward: int
            max_fitness: float
            cfg: PEPACSConfiguration
        """
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
        """
        The Genetic Generalization mechanism. Handles insertion, update and deletion
        of new classifiers in population and possibly other sets.

        Parameters
        ----------
            population: PEPACSClassifiersList
            match_set: PEPACSClassifiersList
            action_set: PEPACSClassifiersList
            p0: Perception
            p1: Perception
            time: int
            cfg: PEPACSConfiguration
        """
        ga.apply(
            PEPACSClassifiersList,
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