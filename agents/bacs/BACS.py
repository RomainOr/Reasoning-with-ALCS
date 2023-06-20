"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common.Agent import Agent, TrialMetrics
from agents.common.classifier_components.BaseClassifier import BaseClassifier
from agents.common.mechanisms.action_selection import choose_classifier

from agents.bacs.BACSConfiguration import BACSConfiguration
from agents.bacs.BACSClassifiersList import BACSClassifiersList


class BACS(Agent):
    """
    Represents a BACS agent
    """

    def __init__(
            self,
            cfg: BACSConfiguration,
            population: BACSClassifiersList=None
        ) -> None:
        super().__init__(
            cfg=cfg,
            population=population or BACSClassifiersList(),
            seed=cfg.seed
        )


    def _run_trial_explore(
            self,
            env,
            time,
            current_trial=None
        ) -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state, _info = env.reset()
        state = self.cfg.environment_adapter.to_genotype(env, raw_state)
        total_reward = 0
        done = False

        # For action chunking
        t_2_activated_classifier = None
        t_1_activated_classifier = None

        # For applying alp on behavioral set
        is_behavioral_sequence = False

        while not done:
            
            # Creation of the matching set
            match_set, best_fitness = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                if is_behavioral_sequence:
                    BACSClassifiersList.apply_alp_behavioral_sequence(
                        self.population,
                        match_set,
                        action_set,
                        prev_state,
                        state,
                        time + steps
                    )
                else:
                    BACSClassifiersList.apply_alp(
                        self.population,
                        match_set,
                        action_set,
                        t_2_activated_classifier,
                        prev_state,
                        t_1_activated_classifier.action,
                        state,
                        time + steps,
                        self.cfg
                    )
                BACSClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    best_fitness,
                    self.cfg
                )
                BACSClassifiersList.apply_ga(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    state,
                    time + steps,
                    self.cfg
                )

            # Choose classifier
            is_behavioral_sequence = False
            action_classifier = choose_classifier(match_set, self.cfg)
            #Record last activated classifier
            t_2_activated_classifier = t_1_activated_classifier
            t_1_activated_classifier = action_classifier
            # Create action set
            action_set = match_set.form_action_set(action_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(env, action_classifier.action)
            # Do the action
            prev_state = state
            raw_state, last_reward, terminated, truncated, _info = env.step(iaction)
            done = terminated or truncated
            total_reward += last_reward
            state = self.cfg.environment_adapter.to_genotype(env, raw_state)

            if done and action_classifier.behavioral_sequence:
                action_set = match_set.form_action_set(
                    BaseClassifier(action=action_classifier.action, cfg=self.cfg)
                )
            
            # Enter the if condition only if we have chosen a behavioral classifier
            if not done and action_classifier.behavioral_sequence:
                bseq_rescue = []
                is_behavioral_sequence = True
                # Initialize the message list usefull to decrease quality of classifiers containing looping sequences
                message_list = [prev_state, state]
                for act in action_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(env, act)
                    raw_state, last_reward, terminated, truncated, _info = env.step(iaction)
                    done = terminated or truncated
                    total_reward += last_reward
                    bseq_rescue.append(act)
                    state = self.cfg.environment_adapter.to_genotype(env, raw_state)
                    if state in message_list:
                        for cl in action_set:    
                            cl.decrease_quality()
                    else:
                        message_list.append(state)
                    if done:
                        action_set = match_set.form_action_set(
                            BaseClassifier(
                                action=action_classifier.action,
                                behavioral_sequence=bseq_rescue,
                                cfg=self.cfg
                            )
                        )
                        break
                    steps += 1

            if done:
                # Apply algorithms
                if is_behavioral_sequence:
                    BACSClassifiersList.apply_alp_behavioral_sequence(
                        self.population,
                        match_set,
                        action_set,
                        prev_state,
                        state,
                        time + steps
                    )
                else:
                    BACSClassifiersList.apply_alp(
                        self.population,
                        match_set,
                        action_set,
                        t_2_activated_classifier,
                        prev_state,
                        t_1_activated_classifier.action,
                        state,
                        time + steps,
                        self.cfg
                    )
                BACSClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    0.,
                    self.cfg
                )
                BACSClassifiersList.apply_ga(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    state,
                    time + steps,
                    self.cfg
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
        raw_state, _info = env.reset()
        state = self.cfg.environment_adapter.to_genotype(env, raw_state)
        last_reward = 0
        done = False

        while not done:

            # Compute in one run the matching set and the best matching fitness
            match_set, best_fitness = self.population.form_match_set(state)

            if steps > 0:
                BACSClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    best_fitness,
                    self.cfg
                )

            # Choose classifier
            best_classifier = choose_classifier(match_set, self.cfg)
            # Create action set
            action_set = match_set.form_action_set(best_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(env, best_classifier.action)
            # Do the action
            raw_state, last_reward, terminated, truncated, _info = env.step(iaction)
            done = terminated or truncated
            state = self.cfg.environment_adapter.to_genotype(env, raw_state)

            if done and best_classifier.behavioral_sequence:
                action_set = match_set.form_action_set(
                    BaseClassifier(action=best_classifier.action, cfg=self.cfg)
                )
            
            # Enter the if condition only if we have chosen a behavioral classifier
            if not done and best_classifier.behavioral_sequence :
                bseq_rescue = []
                for act in best_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(env, act)
                    raw_state, last_reward, terminated, truncated, _info = env.step(iaction)
                    done = terminated or truncated
                    bseq_rescue.append(act)
                    state = self.cfg.environment_adapter.to_genotype(env, raw_state)
                    if done:
                        action_set = match_set.form_action_set(
                            BaseClassifier(
                                action=best_classifier.action,
                                behavioral_sequence=bseq_rescue,
                                cfg=self.cfg
                            )
                        )
                        break
                    steps += 1

            if done:
                # Apply algorithms
                BACSClassifiersList.apply_reinforcement_learning(
                    action_set, 
                    last_reward, 
                    0.,
                    self.cfg
                )

            steps += 1

        return TrialMetrics(steps, last_reward)
