"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import List, Tuple

from bacs import Perception
from bacs.agents.Agent import Agent, TrialMetrics
from bacs.agents.bacs import Classifier, ClassifiersList, Configuration
from bacs.agents.bacs.Condition import Condition
from bacs.agents.bacs.Effect import Effect
from bacs.agents.bacs.components.subsumption_bacs import does_subsume, find_subsumers
from bacs.agents.bacs.components.action_selection_bacs import choose_classifier

class BACS(Agent):

    def __init__(self,
                 cfg: Configuration,
                 population: ClassifiersList=None) -> None:
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
            if to_keep and cl.behavioral_sequence is not None and \
                (not cl.is_experienced() or not cl.is_reliable()):
                to_keep = False
            if to_keep and cl.behavioral_sequence is not None and \
                not cl.does_anticipate_change() and len(cl.effect)==1:
                to_keep = False
            if to_keep:
                classifiers_to_keep.append(cl)
        self.population = ClassifiersList(*classifiers_to_keep)


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

        # For action chunking
        t_2_activated_classifier = None
        t_1_activated_classifier = None

        # For applying alp on behavioral set
        is_behavioral_sequence = False

        while not done:
            
            # Creation of the matching set
            match_set, _, best_fitness = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                if is_behavioral_sequence:
                    ClassifiersList.apply_alp_behavioral_sequence(
                        self.population,
                        match_set,
                        action_set,
                        prev_state,
                        state,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg
                    )
                else:
                    ClassifiersList.apply_alp(
                        self.population,
                        match_set,
                        action_set,
                        prev_state,
                        t_1_activated_classifier.action,
                        state,
                        t_2_activated_classifier,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg
                    )
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    best_fitness,
                    self.cfg.beta_rl,
                    self.cfg.gamma
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
            is_behavioral_sequence = False
            # Choose classifier
            action_classifier = choose_classifier(match_set, self.cfg, self.cfg.epsilon)
            #Record last activated classifier
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
                is_behavioral_sequence = True
                # Initialize the message list usefull to decrease quality of classifiers containing looping sequences
                #message_list = [state]
                for act in action_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(act)
                    raw_state, last_reward, done, _ = env.step(iaction)
                    state = self.cfg.environment_adapter.to_genotype(raw_state)
                    #if state in message_list:
                    #    for cl in action_set:    
                    #        cl.decrease_quality()
                    #else:
                    #    message_list.append(state)
                    if done:
                        break
                    steps += 1

            if done:
                # Apply algorithms
                if is_behavioral_sequence:
                    ClassifiersList.apply_alp_behavioral_sequence(
                        self.population,
                        ClassifiersList(),
                        action_set,
                        prev_state,
                        state,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg)
                else:
                    ClassifiersList.apply_alp(
                        self.population,
                        ClassifiersList(),
                        action_set,
                        prev_state,
                        action,
                        state,
                        t_2_activated_classifier,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg)
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    0,
                    self.cfg.beta_rl,
                    self.cfg.gamma)
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
                    action_set,
                    last_reward,
                    best_fitness,
                    self.cfg.beta_rl,
                    self.cfg.gamma)

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
                # Apply algorithms
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta_rl, self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, last_reward)
