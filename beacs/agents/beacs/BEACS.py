"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import List, Tuple

from beacs import Perception
from beacs.agents.Agent import Agent, TrialMetrics
from beacs.agents.beacs import Classifier, ClassifiersList, Configuration
from beacs.agents.beacs.Condition import Condition
from beacs.agents.beacs.Effect import Effect
from beacs.agents.beacs.components.action_selection import choose_classifier

class BEACS(Agent):

    def __init__(self,
            cfg: Configuration,
            population: ClassifiersList=None
            ) -> None:
        self.cfg = cfg
        self.population = population or ClassifiersList()
        self.pai_states_memory = []

    def get_population(self)-> ClassifiersList:
        return self.population

    def duplicate_population(self)-> ClassifiersList:
        duplicate_population = []
        for cl in self.population:
            cl_copy = Classifier.copy_from(cl, 0)
            cl_copy.num = cl.num
            cl_copy.exp = cl.exp
            cl_copy.tga = cl.tga
            cl_copy.tbseq = cl.tbseq
            cl_copy.talp = cl.talp
            duplicate_population.append(cl_copy)
        return ClassifiersList(*duplicate_population)

    def get_cfg(self)-> Configuration:
        return self.cfg

    def get_pai_states_memory(self):
        return self.pai_states_memory

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
                not cl.is_experienced():
                to_keep = False
            if to_keep and cl.behavioral_sequence is not None and \
                not cl.does_anticipate_change() and len(cl.effect)==1:
                to_keep = False
            if to_keep:
                classifiers_to_keep.append(cl)
        self.population = ClassifiersList(*classifiers_to_keep)


    def _run_trial_explore(
            self,
            env,
            time,
            current_trial=None
        ) -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(env, raw_state)
        last_reward = 0
        total_reward = 0
        prev_state = Perception.empty()
        t_2_match_set = ClassifiersList()
        t_1_match_set = ClassifiersList()
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
                    t_2_match_set,
                    t_1_match_set,
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
                    prev_state,
                    state,
                    self.cfg.theta_ga,
                    self.cfg.mu,
                    self.cfg.chi,
                    self.cfg.theta_as
                )

            # Record the previous match set
            t_2_match_set = t_1_match_set
            t_1_match_set = match_set
            # Choose classifier
            action_classifier = choose_classifier(match_set, self.cfg, self.cfg.epsilon)
            # Tmp : Mountaincar -> epsilon degréssif : max(0.01, self.cfg.epsilon-current_trial/10000)
            # Record last activated classifier
            t_2_activated_classifier = t_1_activated_classifier
            t_1_activated_classifier = action_classifier
            # Create action set
            action_set = match_set.form_action_set(action_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(env, action_classifier.action)
            # Do the action
            prev_state = state
            raw_state, last_reward, done, _ = env.step(iaction)
            total_reward += last_reward
            state = self.cfg.environment_adapter.to_genotype(env, raw_state)
            
            if done and action_classifier.behavioral_sequence:
                action_set = match_set.form_action_set(Classifier(action=action_classifier.action, cfg=self.cfg))

            # Enter the if condition only if we have chosen a behavioral classifier
            if not done and action_classifier.behavioral_sequence :
                bseq_rescue = []
                # Initialize the message list usefull to decrease quality of classifiers containing looping sequences
                for act in action_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(env, act)
                    raw_state, last_reward, done, _ = env.step(iaction)
                    bseq_rescue.append(act)
                    total_reward += last_reward
                    state = self.cfg.environment_adapter.to_genotype(env, raw_state)
                    if done:
                        action_set = match_set.form_action_set(Classifier(action=action_classifier.action, behavioral_sequence=bseq_rescue, cfg=self.cfg))
                        break
                    steps += 1


            if done:
                ClassifiersList.apply_alp(
                    self.population,
                    t_2_match_set,
                    t_1_match_set,
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
                    prev_state,
                    state,
                    self.cfg.theta_ga,
                    self.cfg.mu,
                    self.cfg.chi,
                    self.cfg.theta_as
                )

            steps += 1
        return TrialMetrics(steps, total_reward)

    def _run_trial_exploit(
            self,
            env,
            time,
            current_trial=None
        ) -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(env, raw_state)
        last_reward = 0
        total_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:

            # Compute in one run the matching set, the best matching classifier and the best matching fitness associated to the previous classifier
            match_set, _, max_fitness_ra, max_fitness_rb = self.population.form_match_set(state)

            if steps > 0:
                # Apply algorithms
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, max_fitness_ra, max_fitness_rb, self.cfg.beta_rl, self.cfg.gamma
                )

            # Choose classifier
            best_classifier = choose_classifier(match_set, self.cfg, self.cfg.epsilon)
            # Create action set
            action_set = match_set.form_action_set(best_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(env, best_classifier.action)
            # Do the action
            raw_state, last_reward, done, _ = env.step(iaction)
            total_reward += last_reward
            state = self.cfg.environment_adapter.to_genotype(env, raw_state)

            # Enter the if condition only if we have chosen a behavioral classifier
            if not done and best_classifier.behavioral_sequence :
                for act in best_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(env, act)
                    raw_state, last_reward, done, _ = env.step(iaction)
                    total_reward += last_reward
                    state = self.cfg.environment_adapter.to_genotype(env, raw_state)
                    if done:
                        break
                    steps += 1

            if done:
                # Apply algorithms
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0., 0., self.cfg.beta_rl, self.cfg.gamma
                )

            steps += 1

        return TrialMetrics(steps, total_reward)
