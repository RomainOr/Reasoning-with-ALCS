from collections import namedtuple
from typing import Callable, List, Tuple

TrialMetrics = namedtuple('TrialMetrics', ['steps', 'reward'])


class Agent:

    def _run_trial_explore(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError()

    def _run_trial_exploit(self, env, trials, current_trial) -> TrialMetrics:
        raise NotImplementedError()

    def get_population(self):
        raise NotImplementedError()

    def get_cfg(self):
        raise NotImplementedError()

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
