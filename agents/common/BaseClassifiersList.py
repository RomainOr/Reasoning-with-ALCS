"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations
from operator import attrgetter
from itertools import chain
from typing import List

from agents.common.BaseConfiguration import BaseConfiguration
from agents.common.Perception import Perception
from agents.common.TypedList import TypedList
from agents.common.classifier_components.BaseClassifier import BaseClassifier


class BaseClassifiersList(TypedList):
    """
    Represents overall population, match/action sets
    """

    def __init__(self, oktypes = (BaseClassifier, ), *args) -> None:
        super().__init__(oktypes, *args)


    def form_match_set(
            self,
            situation: Perception
        ):
        """
        Builds theBaseClassifiersList from the whole population with all classifiers whose condition
        matches the current situation.

        Parameters
        ----------
        situation: Perception
            Current perception

        Returns
        ----------
        The whole set of matching classifiers
        """
        best_fitness = 0.0
        matching = []
        for cl in self:
            if cl.does_match(situation):
                matching.append(cl)
                if cl.does_anticipate_change() and cl.fitness > best_fitness:
                    best_fitness = cl.fitness
        return type(self)(*matching), best_fitness



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
        return type(self)(*matching)
    

    def find_best_classifier(
            self,
            situation: Perception
        ):
        return max([cl for cl in self if cl.does_match(situation) and cl.does_anticipate_change()],key=attrgetter('fitness'),default=None)


    def expand(self) -> List[BaseClassifier]:
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
    def merge_newly_built_classifiers(
            new_list:BaseClassifiersList,
            population:BaseClassifiersList,
            match_set:BaseClassifiersList,
            action_set:BaseClassifiersList,
            p0: Perception,
            p1: Perception
        ) -> None:
        """
        Merge classifiers from new_list into self and population

        Parameters
        ----------
        new_list
        population
        match_set
        action_set
        p0: Perception
        p1: Perception
        """
        population.extend(new_list)
        if match_set:
            new_matching = [cl for cl in new_list if cl.does_match(p1)]
            match_set.extend(new_matching)
        if action_set:
            new_action_cls = [cl for cl in new_list if cl.does_match(p0) and cl.action == action_set[0].action and cl.behavioral_sequence == action_set[0].behavioral_sequence]
            action_set.extend(new_action_cls)


    def __str__(self):
        return "\n".join(str(classifier)
            for classifier
            in sorted(self, key=lambda cl: -cl.fitness))


    @staticmethod
    def apply_alp(
            population:BaseClassifiersList,
            t_2_match_set:BaseClassifiersList,
            t_1_match_set:BaseClassifiersList,
            match_set:BaseClassifiersList,
            action_set:BaseClassifiersList,
            penultimate_classifier: BaseClassifier,
            p0: Perception,
            action: int,
            p1: Perception,
            time: int,
            pai_states_memory,
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
        raise NotImplementedError("Subclasses should implement this method.")


    @staticmethod
    def apply_reinforcement_learning(
            action_set:BaseClassifiersList,
            reward: int,
            max_fitness_ra: float,
            max_fitness_rb: float,
            beta_rl: float,
            gamma: float
        ) -> None:
        raise NotImplementedError("Subclasses should implement this method.")


    @staticmethod
    def apply_ga(
            time: int,
            population:BaseClassifiersList,
            match_set:BaseClassifiersList,
            action_set:BaseClassifiersList,
            p0: Perception,
            p1: Perception,
            theta_ga: int,
            mu: float,
            chi: float,
            theta_as: int
        ) -> None:
        raise NotImplementedError("Subclasses should implement this method.")
