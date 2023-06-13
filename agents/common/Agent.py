"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from collections import namedtuple
from typing import Callable, List, Tuple

from agents.common.BaseClassifiersList import BaseClassifiersList
from agents.common.BaseConfiguration import BaseConfiguration
from agents.common.RandomNumberGenerator import RandomNumberGenerator

TrialMetrics = namedtuple('TrialMetrics', ['steps', 'reward'])


class Agent:

    def __init__(self,
            cfg: BaseConfiguration,
            population: BaseClassifiersList,
            seed):
        self.cfg = cfg
        self.population = population
        RandomNumberGenerator.seed(seed)


    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError("Subclasses should implement this method.")


    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError("Subclasses should implement this method.")


    def get_population(self):
        raise NotImplementedError("Subclasses should implement this method.")


    def get_cfg(self):
        return self.cfg


    def get_population(self)-> BaseClassifiersList:
        return self.population


    def duplicate_population(self)-> BaseClassifiersList:
        raise NotImplementedError("Subclasses should implement this method.")


    def apply_CRACS(self):
        # Removing subsumed classifiers and unwanted behavioral classifiers
        classifiers_to_keep = []
        for cl in self.population:
            to_keep = True
            for other in self.population:
                if cl != other and other.subsumes(cl):
                    cl.average_fitnesses_from_other_cl(other)
                    to_keep = False
                    break
            if to_keep and cl.behavioral_sequence is not None and \
                not cl.is_experienced():
                to_keep = False
            if to_keep and cl.behavioral_sequence is not None and \
                not cl.does_anticipate_change() and len(cl.effect)==1:
                to_keep = False
            if to_keep:
                classifiers_to_keep.append(cl)
        idx = 0
        population_length = len(self.population)
        while(idx < population_length):
            if self.population[idx] not in classifiers_to_keep:
                self.population.safe_remove(self.population[idx])
                idx -= 1
                population_length -= 1
            idx += 1


    def explore(self, env, trials) -> Tuple:
        """
        Explores the environment in given set of trials.

        Parameters
        ----------
        env
            environment
        trials
            number of trials

        Returns
        -------
        Tuple
            population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_explore)


    def exploit(self, env, trials) -> Tuple:
        """
        Exploits the environments in given set of trials (always executing
        best possible action - no exploration).

        Parameters
        ----------
        env
            environment
        trials
            number of trials

        Returns
        -------
        Tuple
            population of classifiers and metrics
        """
        return self._evaluate(env, trials, self._run_trial_exploit)


    def _evaluate(self, env, max_trials: int, func: Callable) -> Tuple:
        """
        Runs the classifier in desired strategy (see `func`) and collects
        metrics.

        Parameters
        ----------
        env:
            OpenAI Gym environment
        max_trials: int
            maximum number of trials
        func: Callable
            Function accepting three parameters: env, steps already made,
             current trial

        Returns
        -------
        tuple
            population of classifiers and metrics
        """

        def _basic_metrics(
            trial: int,
            steps: int, 
            reward: int
            ) -> dict:
            return {
                'trial': trial,
                'steps_in_trial': steps,
                'reward': reward
            }

        current_trial = 1
        steps = 0

        metrics: List = []
        while current_trial <= max_trials:
            steps_in_trial, reward = func(env, steps, current_trial)
            steps += steps_in_trial

            if current_trial % self.get_cfg().metrics_trial_frequency == 0:
                m = _basic_metrics(current_trial, steps_in_trial, reward)
                user_metrics = self.get_cfg().user_metrics_collector_fcn
                if user_metrics is not None:
                    m.update(user_metrics(self.get_population(), env))
                metrics.append(m)

            current_trial += 1

        return self.get_population(), metrics
