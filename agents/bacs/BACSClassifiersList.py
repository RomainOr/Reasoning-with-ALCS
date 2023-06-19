"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from typing import Optional

import agents.common.mechanisms.alp as alp_common
import agents.common.mechanisms.genetic_algorithms as ga
from agents.common.BaseClassifiersList import BaseClassifiersList
from agents.common.Perception import Perception
from agents.common.classifier_components.BaseClassifier import BaseClassifier
from agents.common.mechanisms.reinforcement_learning import update_classifier_q_learning

from agents.bacs.BACSConfiguration import BACSConfiguration
import agents.bacs.mechanisms.alp as alp_bacs


class BACSClassifiersList(BaseClassifiersList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args) -> None:
        super().__init__((BaseClassifier, ), *args)


    @staticmethod
    def apply_alp(
            population: BACSClassifiersList,
            match_set: BACSClassifiersList,
            action_set: BACSClassifiersList,
            last_activated_classifier: BaseClassifier,
            p0: Perception,
            action: int,
            p1: Perception,
            time: int,
            cfg: BACSConfiguration
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
        last_activated_classifier: BaseClassifier
        time: int
        cfg: BaseConfiguration
        """
        new_list = BACSClassifiersList()
        new_cl: Optional[BaseClassifier] = None
        was_expected_case = False

        idx = 0
        action_set_length = 0
        if action_set: action_set_length = len(action_set)

        while(idx < action_set_length):
            cl = action_set[idx]
            cl.increase_experience()
            cl.set_alp_timestamp(time)

            if cl.does_anticipate_correctly(p0, p1):
                new_cl = alp_bacs.expected_case(last_activated_classifier, cl, p0, p1, time)
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
                if new_cl.behavioral_sequence:
                    alp_common.add_classifier(new_cl, population, new_list)
                else:
                    alp_common.add_classifier(new_cl, action_set, new_list)

        # No classifier anticipated correctly - generate new one
        if not was_expected_case:
            new_cl = alp_common.cover(BaseClassifier, p0, action, p1, time, cfg)
            alp_common.add_classifier(new_cl, action_set, new_list)

        # Merge classifiers from new_list into self and population
        BACSClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)


    @staticmethod
    def apply_alp_behavioral_sequence(
            population: BACSClassifiersList,
            match_set: BACSClassifiersList,
            action_set: BACSClassifiersList,
            p0: Perception,
            p1: Perception,
            time: int,
            cfg: BACSConfiguration
        ) -> None:
        """
        The Anticipatory Learning Process when a behavioral sequence has been executed

        Parameters
        ----------
        population
        match_set
        action_set
        p0: Perception
        p1: Perception
        time: int
        cfg: BaseConfiguration
        """
        new_list = BACSClassifiersList()
        new_cl: Optional[BaseClassifier] = None

        idx = 0
        action_set_length = 0
        if action_set: action_set_length = len(action_set)

        while(idx < action_set_length):
            cl = action_set[idx]
            cl.increase_experience()
            cl.set_alp_timestamp(time)

            # Useless case
            if (p0 == p1):
                cl.decrease_quality()
            # Expected case
            elif cl.does_anticipate_correctly(p0, p1):
                new_cl = alp_common.expected_case(cl, p0, time, p1)
            # Unexpected case
            else:
                new_cl = alp_common.unexpected_case(cl, p0, p1, time)

            if new_cl is not None:
                new_cl.tga = time
                alp_common.add_classifier(new_cl, action_set, new_list)

            # Quality Anticipation check
            if cl.is_inadequate():
                # Removes classifier from population, match set
                # and current list
                lists = [x for x in [population, match_set, action_set] if x]
                for lst in lists:
                    lst.safe_remove(cl)
                idx -= 1
                action_set_length -= 1
            idx += 1

        # Merge classifiers from new_list into action_set and population
        BACSClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)


    @staticmethod
    def apply_reinforcement_learning(
            action_set: BACSClassifiersList,
            reward: int,
            max_fitness: float,
            cfg: BACSConfiguration
        ) -> None:
        for cl in action_set:
            update_classifier_q_learning(cl, reward, max_fitness, cfg.beta_rl, cfg.gamma)


    @staticmethod
    def apply_ga(
            population: BACSClassifiersList,
            match_set: BACSClassifiersList,
            action_set: BACSClassifiersList,
            p0: Perception,
            p1: Perception,
            time: int,
            cfg: BACSConfiguration
        ) -> None:
        if action_set is None or not action_set:
            return False
        
        if action_set[0].behavioral_sequence:
            return False

        ga.apply(
            BACSClassifiersList,
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
