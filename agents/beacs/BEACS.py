"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common.Agent import Agent, TrialMetrics
from agents.common.mechanisms.action_selection import choose_classifier

from agents.beacs.BEACSClassifiersList import BEACSClassifiersList
from agents.beacs.BEACSConfiguration import BEACSConfiguration
from agents.beacs.classifier_components.BEACSClassifier import BEACSClassifier


class BEACS(Agent):
    """
    Represents a BEACS agent
    """

    def __init__(self,
            cfg: BEACSConfiguration,
            population: BEACSClassifiersList=None
            ) -> None:
        super().__init__(
            cfg=cfg,
            population=population or BEACSClassifiersList(),
            seed=cfg.seed
        )
        self.pai_states_memory = []


    def get_pai_states_memory(self) -> list:
        """
        Return the list of states perceived as related to the Perceptual Aliasing Issue.

        Returns
        ----------
        list
            The list of states related to the PAI
        """
        return self.pai_states_memory


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
        t_2_match_set = BEACSClassifiersList()
        t_1_match_set = BEACSClassifiersList()
        t_2_activated_classifier = None
        t_1_activated_classifier = None

        while not done:
            
            # Creation of the matching set
            match_set, max_fitness_r, max_fitness_r_bis = self.population.form_match_set(state)

            # Apply learning in the last action set
            if steps > 0:
                BEACSClassifiersList.apply_alp(
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
                BEACSClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    max_fitness_r,
                    max_fitness_r_bis, 
                    self.cfg
                )
                BEACSClassifiersList.apply_ga(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    state,
                    time + steps,
                    self.cfg
                )

            # Record the previous match set
            t_2_match_set = t_1_match_set
            t_1_match_set = match_set
            # Choose classifier
            action_classifier = choose_classifier(match_set, self.cfg)
            # Record last activated classifier
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
                action_set = match_set.form_action_set(BEACSClassifier(action=action_classifier.action, cfg=self.cfg))

            # Enter the if condition only if we have chosen a behavioral classifier
            if not done and action_classifier.behavioral_sequence :
                bseq_rescue = []
                # Initialize the message list usefull to decrease quality of classifiers containing looping sequences
                for act in action_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(env, act)
                    raw_state, last_reward, terminated, truncated, _info = env.step(iaction)
                    done = terminated or truncated
                    bseq_rescue.append(act)
                    total_reward += last_reward
                    state = self.cfg.environment_adapter.to_genotype(env, raw_state)
                    if done:
                        action_set = match_set.form_action_set(BEACSClassifier(action=action_classifier.action, behavioral_sequence=bseq_rescue, cfg=self.cfg))
                        break
                    steps += 1

            if done:
                BEACSClassifiersList.apply_alp(
                    self.population,
                    t_2_match_set,
                    t_1_match_set,
                    BEACSClassifiersList(),
                    action_set,
                    t_2_activated_classifier,
                    prev_state,
                    t_1_activated_classifier.action,
                    state,
                    time + steps,
                    self.pai_states_memory,
                    self.cfg
                )
                BEACSClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward, 
                    0.,
                    0.,
                    self.cfg
                )
                BEACSClassifiersList.apply_ga(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    state,
                    time + steps,
                    self.cfg
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
        raw_state, _info = env.reset()
        state = self.cfg.environment_adapter.to_genotype(env, raw_state)
        total_reward = 0
        done = False

        while not done:

            # Compute in one run the matching set, the best matching classifier and the best matching fitness associated to the previous classifier
            match_set, max_fitness_r, max_fitness_r_bis = self.population.form_match_set(state)

            if steps > 0:
                # Apply algorithms
                BEACSClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    max_fitness_r,
                    max_fitness_r_bis, 
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
            total_reward += last_reward
            state = self.cfg.environment_adapter.to_genotype(env, raw_state)

            # Enter the if condition only if we have chosen a behavioral classifier
            if not done and best_classifier.behavioral_sequence :
                for act in best_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(env, act)
                    raw_state, last_reward, terminated, truncated, _info = env.step(iaction)
                    done = terminated or truncated
                    total_reward += last_reward
                    state = self.cfg.environment_adapter.to_genotype(env, raw_state)
                    if done:
                        break
                    steps += 1

            if done:
                # Apply algorithms
                BEACSClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward, 
                    0.,
                    0.,
                    self.cfg
                )

            steps += 1

        return TrialMetrics(steps, total_reward)
