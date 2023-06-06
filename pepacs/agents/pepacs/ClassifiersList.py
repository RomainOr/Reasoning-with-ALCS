"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from __future__ import annotations

from itertools import chain
from typing import Optional, List

import pepacs.agents.pepacs.components.alp as alp
import pepacs.agents.pepacs.components.genetic_algorithms as ga
import pepacs.agents.pepacs.components.reinforcement_learning as rl
from pepacs import Perception, TypedList, RandomNumberGenerator
from pepacs.agents.pepacs import Configuration
from pepacs.agents.pepacs.classifier_components import Classifier

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
    def merge_newly_built_classifiers(
            new_list: ClassifiersList,
            population: ClassifiersList,
            match_set: ClassifiersList,
            action_set: ClassifiersList,
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
            new_action_cls = [cl for cl in new_list if cl.does_match(p0) and cl.action == action_set[0].action]
            action_set.extend(new_action_cls)



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
        for candidate in candidates:
            candidates2 = [cl for cl in candidates if candidate != cl and cl.mark == candidate.mark]
            if len(candidates2) > 0:
                merger = RandomNumberGenerator.choice(candidates2)
                new_classifier = candidate.merge_with(merger, previous_situation, time)
                if new_classifier is not None:
                    alp.add_classifier(new_classifier, action_set, new_list, cfg.theta_exp)
        return new_list


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
                alp.add_classifier(new_cl, action_set, new_list, theta_exp)

        if cfg.do_pep:
            ClassifiersList.apply_enhanced_effect_part_check(action_set, new_list, p0, time, cfg)

        # No classifier anticipated correctly - generate new one
        if not was_expected_case:
            new_cl = alp.cover(p0, action, p1, time, cfg)
            alp.add_classifier(new_cl, action_set, new_list, theta_exp)

        # Merge classifiers from new_list into self and population
        ClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)


    @staticmethod
    def apply_reinforcement_learning(
            action_set: ClassifiersList,
            reward: int,
            max_fitness: float,
            beta_rl: float,
            gamma: float
        ) -> None:
        for cl in action_set:
            rl.update_classifier(cl, reward, max_fitness, beta_rl, gamma)


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

            child1 = Classifier.copy_from(parent1, p1, time, False)
            child2 = Classifier.copy_from(parent2, p1, time, False)

            # Execute mutation
            ga.generalizing_mutation(child1, mu)
            ga.generalizing_mutation(child2, mu)

            # Execute cross-over
            if RandomNumberGenerator.random() < chi:
                if child1.effect == child2.effect:
                    ga.two_point_crossover(child1, child2)

                    # Update quality and reward
                    child1.q = child2.q = float(sum([child1.q, child2.q]) / 2)
                    child2.r = child2.r = float(sum([child1.r, child2.r]) / 2)

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

            new_list = ClassifiersList()
            # check for subsumers / similar classifiers
            for child in children:
                ga.add_classifier(
                    child,
                    action_set,
                    new_list,
                    theta_exp
                
                )

            # Merge classifiers from new_list into self and population
            ClassifiersList.merge_newly_built_classifiers(new_list, population, match_set, action_set, p0, p1)


    def __str__(self):
        return "\n".join(str(classifier)
            for classifier
            in sorted(self, key=lambda cl: -cl.fitness))
