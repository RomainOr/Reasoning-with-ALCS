"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import List, Tuple

from epeacs import Perception
from epeacs.agents.Agent import Agent, TrialMetrics
from epeacs.agents.epeacs import Classifier, ClassifiersList, Configuration
from epeacs.agents.epeacs.Condition import Condition
from epeacs.agents.epeacs.Effect import Effect
from epeacs.agents.epeacs.components.action_selection import choose_action

class EPEACS(Agent):

    def __init__(self,
            cfg: Configuration,
            population: ClassifiersList=None
            ) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def _run_trial_explore(self, env, time, current_trial=None) \
            -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        action = env.action_space.sample()
        last_reward = 0
        prev_state = Perception.empty()
        action_set = ClassifiersList()
        done = False

        while not done:
            
            # Creation of the matching set
            match_set, _, best_fitness = self.population.form_match_set(state)

            if steps > 0:
                ClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg
                )
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, best_fitness, self.cfg.beta_rl, self.cfg.gamma
                )
                if self.cfg.do_ga:
                    ClassifiersList.apply_ga(
                        time + steps,
                        self.population,
                        match_set,
                        action_set,
                        state,
                        self.cfg.theta_ga,
                        self.cfg.mu,
                        self.cfg.chi,
                        self.cfg.theta_as,
                        self.cfg.theta_exp
                    )
                if self.cfg.do_zip:
                    ClassifiersList.apply_zip(
                        time + steps,
                        self.population,
                        self.cfg.theta_zip,
                        self.cfg.theta_exp
                    )

            action = choose_action(match_set, self.cfg, self.cfg.epsilon)
            # Create action set
            action_set = match_set.form_action_set(action)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            # Do the action
            prev_state = state
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            if done:
                ClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action,
                    state,
                    time + steps,
                    self.cfg.theta_exp,
                    self.cfg
                )
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta_rl, self.cfg.gamma
                )
            if self.cfg.do_ga:
                ClassifiersList.apply_ga(
                    time + steps,
                    self.population,
                    ClassifiersList(),
                    action_set,
                    state,
                    self.cfg.theta_ga,
                    self.cfg.mu,
                    self.cfg.chi,
                    self.cfg.theta_as,
                    self.cfg.theta_exp)

            steps += 1
        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, time=None, current_trial=None) \
            -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        last_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:

            # Compute in one run the matching set, the best matching classifier and the best matching fitness associated to the previous classifier
            match_set, best_classifier, best_fitness = self.population.form_match_set(state)

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, best_fitness, self.cfg.beta_rl, self.cfg.gamma
                )

            # Create action set
            action_set = match_set.form_action_set(best_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(best_classifier.action)
            # Do the action
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            if done:
                # Apply algorithms
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta_rl, self.cfg.gamma
                )

            steps += 1

        return TrialMetrics(steps, last_reward)
