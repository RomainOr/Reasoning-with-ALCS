"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import List, Tuple

from pepacs import Perception
from pepacs.agents.Agent import Agent, TrialMetrics
from pepacs.agents.pepacs import Classifier, ClassifiersList, Configuration
from pepacs.agents.pepacs.Condition import Condition
from pepacs.agents.pepacs.Effect import Effect
from pepacs.agents.pepacs.components.subsumption import does_subsume, find_subsumers
from pepacs.agents.pepacs.components.action_selection import choose_action

class PEPACS(Agent):

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

    def zip_population(
            self,
            does_anticipate_change:bool=False,
            is_reliable:bool=False
        ):
        # Remove multiple occurence of same classifiers
        self.population = ClassifiersList(*list(dict.fromkeys(self.population)))
        # Keep or not classifiers that anticipate changes
        if does_anticipate_change:
            pop = [cl for cl in self.population if cl.does_anticipate_change()]
            self.population = ClassifiersList(*pop)
        # Keep all classifiers or only reliable classifiers
        if is_reliable:
            pop = [cl for cl in self.population if cl.is_reliable()]
            self.population = ClassifiersList(*pop)
        # Removing subsumed classifiers and unwanted behavioral classifiers
        classifiers_to_keep = []
        for cl in self.population:
            to_keep = True
            for other in self.population:
                if cl != other and other.subsumes(cl):
                    to_keep = False
                    break
            #if to_keep and cl.behavioral_sequence is not None and \
            #    (not cl.is_experienced() or not cl.is_reliable()):
            #    to_keep = False
            #if to_keep and cl.behavioral_sequence is not None and \
            #    not cl.does_anticipate_change() and len(cl.effect)==1:
            #    to_keep = False
            if to_keep:
                classifiers_to_keep.append(cl)
        self.population = ClassifiersList(*classifiers_to_keep)


    def _run_trial_explore(self, env, time, current_trial=None) \
            -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(env,raw_state)
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

            action = choose_action(match_set, self.cfg, self.cfg.epsilon)
            # Create action set
            action_set = match_set.form_action_set(action)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(env,action)
            # Do the action
            prev_state = state
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(env,raw_state)

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
        state = self.cfg.environment_adapter.to_genotype(env,raw_state)
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
            action_set = match_set.form_action_set(best_classifier.action)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(env,best_classifier.action)
            # Do the action
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(env,raw_state)

            if done:
                # Apply algorithms
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta_rl, self.cfg.gamma
                )

            steps += 1

        return TrialMetrics(steps, last_reward)
