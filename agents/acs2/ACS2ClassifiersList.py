"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from typing import Optional

import agents.common.mechanisms.alp as alp_common
import agents.common.mechanisms.genetic_algorithms as ga
from agents.common.BaseConfiguration import BaseConfiguration
from agents.common.Perception import Perception
from agents.common.BaseClassifiersList import BaseClassifiersList
from agents.common.classifier_components.BaseClassifier import BaseClassifier
from agents.common.mechanisms.reinforcement_learning import update_classifier_q_learning


class ACS2ClassifiersList(BaseClassifiersList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, *args) -> None:
        super().__init__((BaseClassifier, ),*args)


    @staticmethod
    def apply_alp(
            population: ACS2ClassifiersList,
            match_set: ACS2ClassifiersList,
            action_set: ACS2ClassifiersList,
            p0: Perception,
            action: int,
            p1: Perception,
            time: int,
            cfg: BaseConfiguration
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
        cfg: BaseConfiguration
        """
        new_list = ACS2ClassifiersList()
        new_cl: Optional[BaseClassifier] = None
        was_expected_case = False

        idx = 0
        action_set_length = 0
        if action_set: action_set_length = len(action_set)

        #Main ALP loop on the action set
        while(idx < action_set_length):
            cl = action_set[idx]
            cl.increase_experience()
            cl.set_alp_timestamp(time)

            if cl.does_anticipate_correctly(p0, p1):
                new_cl = alp_common.expected_case(cl, p0, time)
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
                alp_common.add_classifier(new_cl, action_set, new_list)

        # No classifier anticipated correctly - generate new one through covering
        if not was_expected_case:
            new_cl = alp_common.cover(BaseClassifier, p0, action, p1, time, cfg)
            alp_common.add_classifier(new_cl, action_set, new_list)

        # Merge classifiers from new_list into self and population
        ACS2ClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)


    @staticmethod
    def apply_reinforcement_learning(
            action_set: ACS2ClassifiersList,
            reward: int,
            max_fitness: float,
            cfg: BaseConfiguration
        ) -> None:
        for cl in action_set:
            update_classifier_q_learning(cl, reward, max_fitness, cfg.beta_rl, cfg.gamma)


    @staticmethod
    def apply_ga(
            population: ACS2ClassifiersList,
            match_set: ACS2ClassifiersList,
            action_set: ACS2ClassifiersList,
            p0: Perception,
            p1: Perception,
            time: int,
            cfg: BaseConfiguration
        ) -> None:
        ga.apply(
            ACS2ClassifiersList,
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
