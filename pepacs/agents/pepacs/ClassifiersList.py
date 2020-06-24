"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

import random
from itertools import chain
from typing import Optional, List

import pepacs.agents.pepacs.components.alp as alp
import pepacs.agents.pepacs.components.genetic_algorithms as ga
import pepacs.agents.pepacs.components.reinforcement_learning as rl
from pepacs import Perception, TypedList
from pepacs.agents.pepacs import Classifier, Configuration
from pepacs.agents.pepacs.components.add_classifier import add_classifier
from pepacs.agents.pepacs.ProbabilityEnhancedAttribute import ProbabilityEnhancedAttribute

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
                if cl.does_anticipate_change() and cl.fitness > best_fitness:
                    best_classifier = cl
                    best_fitness = cl.fitness
        return ClassifiersList(*matching), best_classifier, best_fitness


    def form_action_set(self, action: int) -> ClassifiersList:
        matching = [cl for cl in self if cl.action == action]
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
        # Create a list of candidates.
        # Every enhanceable classifier is a candidate.
        candidates = [cl for cl in action_set if cl.ee]
        # If there are less than 2 candidates, don't do it
        if len(candidates) < 2:
            return
        for idx, first in enumerate(candidates):
            if idx + 1 < len(candidates):
                for second in candidates[idx+1:]:
                    if first != second and first.mark == second.mark:
                        new_classifier = first.merge_with(second, previous_situation, time)
                        add_classifier(new_classifier, action_set, new_list, cfg.theta_exp)


    @staticmethod
    def apply_alp(
            population: ClassifiersList,
            match_set: ClassifiersList,
            action_set: ClassifiersList,
            p0: Perception,
            action: int,
            p1: Perception,
            time: int,
            theta_exp: int,
            cfg: Configuration
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
        theta_exp
        cfg: Configuration
        """
        new_list = ClassifiersList()
        new_cl: Optional[Classifier] = None
        was_expected_case = False

        idx = 0
        action_set_length = 0
        if action_set: action_set_length = len(action_set)

        while(idx < action_set_length):
            cl = action_set[idx]
            cl.increase_experience()
            cl.set_alp_timestamp(time)

            if cl.does_anticipate_correctly(p0, p1):
                new_cl = alp.expected_case(cl, p0, p1, time)
                was_expected_case = True
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
                new_cl.tga = time
                add_classifier(new_cl, action_set, new_list, theta_exp)

        if cfg.do_pep:
            ClassifiersList.apply_enhanced_effect_part_check(action_set, new_list, p0, time, cfg)

        # No classifier anticipated correctly - generate new one
        if not was_expected_case:
            new_cl = alp.cover(p0, action, p1, time, cfg)
            add_classifier(new_cl, action_set, new_list, theta_exp)

        # Merge classifiers from new_list into self and population
        action_set.extend(new_list)
        population.extend(new_list)

        if match_set is not None:
            new_matching = [cl for cl in new_list if cl.condition.does_match(p1)]
            match_set.extend(new_matching)


    @staticmethod
    def apply_reinforcement_learning(
            action_set: ClassifiersList,
            reward: int,
            p: float,
            beta_rl: float,
            gamma: float
        ) -> None:
        for cl in action_set:
            rl.update_classifier(cl, reward, p, beta_rl, gamma)


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
                    child2.r = child2.r = float(sum([child1.r, child2.r]) / 2)

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
