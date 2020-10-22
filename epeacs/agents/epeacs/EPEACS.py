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
from epeacs.agents.epeacs.components.action_selection import choose_classifier

class EPEACS(Agent):

    def __init__(self,
            cfg: Configuration,
            population: ClassifiersList=None
            ) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()
        self.pai_states_memory = []

    def get_population(self):
        return self.population

    def get_cfg(self):
        return self.cfg

    def get_pai_states_memory(self):
        return self.pai_states_memory

    def zip_population(
            self,
            does_anticipate_change:bool = True,
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


    def _run_trial_explore(
            self,
            env,
            time,
            current_trial=None
        ) -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        last_reward = 0
        prev_state = Perception.empty()
        previous_match_set = ClassifiersList()
        match_set = ClassifiersList()
        action_set = ClassifiersList()
        done = False

        # For action chunking
        t_2_activated_classifier = None
        t_1_activated_classifier = None

        while not done:
            
            # Creation of the matching set
            match_set, _, max_fitness_ra, max_fitness_rb = self.population.form_match_set(state)

            # Apply learning in the last action set
            if steps > 0:
                ClassifiersList.apply_alp(
                    self.population,
                    previous_match_set,
                    match_set,
                    action_set,
                    t_2_activated_classifier,
                    prev_state,
                    t_1_activated_classifier.action,
                    state,
                    time + steps,
                    self.pai_states_memory,
                    self.cfg
                )
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, max_fitness_ra, max_fitness_rb, self.cfg.beta_rl, self.cfg.gamma
                )
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
                    self.cfg.theta_exp,
                    self.cfg.do_ga
                )

            # Record the previous match set
            previous_match_set = match_set
            # Choose classifier
            action_classifier = choose_classifier(match_set, self.cfg, self.cfg.epsilon)
            # Record last activated classifier
            t_2_activated_classifier = t_1_activated_classifier
            t_1_activated_classifier = action_classifier
            # Create action set
            action_set = match_set.form_action_set(action_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(action_classifier.action)
            # Do the action
            prev_state = state
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            # Enter the if condition only if we have chosen a behavioral classifier
            if action_classifier.behavioral_sequence :
                # Initialize the message list usefull to decrease quality of classifiers containing looping sequences
                for act in action_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(act)
                    raw_state, last_reward, done, _ = env.step(iaction)
                    state = self.cfg.environment_adapter.to_genotype(raw_state)
                    if done:
                        break
                    steps += 1


            if done:
                ClassifiersList.apply_alp(
                    self.population,
                    previous_match_set,
                    ClassifiersList(),
                    action_set,
                    t_2_activated_classifier,
                    prev_state,
                    t_1_activated_classifier.action,
                    state,
                    time + steps,
                    self.pai_states_memory,
                    self.cfg
                )
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0., 0., self.cfg.beta_rl, self.cfg.gamma
                )
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
                    self.cfg.theta_exp,
                    self.cfg.do_ga
                )

            steps += 1
        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(
            self,
            env,
            time,
            current_trial=None
        ) -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        last_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:

            # Compute in one run the matching set, the best matching classifier and the best matching fitness associated to the previous classifier
            match_set, best_classifier, max_fitness_ra, max_fitness_rb = self.population.form_match_set(state)

            if steps > 0:
                # TODO : Update experience of classifier to get clues about their usage
                for cl in action_set:
                    cl.increase_experience()
                # Apply algorithms
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, max_fitness_ra, max_fitness_rb, self.cfg.beta_rl, self.cfg.gamma
                )

            # Create action set
            action_set = match_set.form_action_set(best_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(best_classifier.action)
            # Do the action
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            # Enter the if condition only if we have chosen a behavioral classifier
            if best_classifier.behavioral_sequence :
                for act in best_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(act)
                    raw_state, last_reward, done, _ = env.step(iaction)
                    state = self.cfg.environment_adapter.to_genotype(raw_state)
                    if done:
                        break
                    steps += 1

            if done:
                # TODO : Update experience of classifier to get clues about their usage
                for cl in action_set:
                    cl.increase_experience()
                # Apply algorithms
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0., 0., self.cfg.beta_rl, self.cfg.gamma
                )

            steps += 1

        return TrialMetrics(steps, last_reward)
