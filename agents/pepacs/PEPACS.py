"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from agents.common.Perception import Perception
from agents.common.Agent import Agent, TrialMetrics
from agents.common.mechanisms.action_selection import choose_classifier

from agents.pepacs.PEPACSConfiguration import PEPACSConfiguration
from agents.pepacs.PEPACSClassifiersList import PEPACSClassifiersList
from agents.pepacs.classifier_components.PEPACSClassifier import PEPACSClassifier


class PEPACS(Agent):

    def __init__(self,
            cfg: PEPACSConfiguration,
            population: PEPACSClassifiersList=None
            ) -> None:
        super().__init__(
            cfg=cfg,
            population=population or PEPACSClassifiersList(),
            seed=cfg.seed
        )


    def duplicate_population(self)-> PEPACSClassifiersList:
        duplicate_population = []
        for cl in self.population:
            cl_copy = PEPACSClassifier.copy_from(old_cl=cl, time=0, Perception=None)
            cl_copy.num = cl.num
            cl_copy.exp = cl.exp
            cl_copy.tga = cl.tga
            cl_copy.talp = cl.talp
            duplicate_population.append(cl_copy)
        return PEPACSClassifiersList(*duplicate_population)
    
    
    def _run_trial_explore(
            self,
            env,
            time,
            current_trial=None
        ) -> TrialMetrics:

        # Initial conditions
        steps = 0
        raw_state, _info = env.reset()
        state = self.cfg.environment_adapter.to_genotype(env,raw_state)
        last_reward = 0
        total_reward = 0
        prev_state = Perception.empty()
        match_set = PEPACSClassifiersList()
        action_set = PEPACSClassifiersList()
        done = False

        while not done:
            
            # Creation of the matching set
            match_set, _, best_fitness = self.population.form_match_set(state)

            if steps > 0:
                PEPACSClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action_classifier.action,
                    state,
                    time + steps,
                    self.cfg
                )
                PEPACSClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    best_fitness,
                    self.cfg
                )
                PEPACSClassifiersList.apply_ga(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    state,
                    time + steps,
                    self.cfg
                )

            # Choose classifier
            action_classifier = choose_classifier(match_set, self.cfg)
            # Create action set
            action_set = match_set.form_action_set(action_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(env,action_classifier.action)
            # Do the action
            prev_state = state
            raw_state, last_reward, terminated, truncated, _info = env.step(iaction)
            done = terminated or truncated
            total_reward += last_reward
            state = self.cfg.environment_adapter.to_genotype(env,raw_state)

            if done:
                PEPACSClassifiersList.apply_alp(
                    self.population,
                    match_set,
                    action_set,
                    prev_state,
                    action_classifier.action,
                    state,
                    time + steps,
                    self.cfg
                )
                PEPACSClassifiersList.apply_reinforcement_learning(
                    action_set, 
                    last_reward, 
                    0.,
                    self.cfg
                )
                PEPACSClassifiersList.apply_ga(
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
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(env,raw_state)
        last_reward = 0
        total_reward = 0
        action_set = PEPACSClassifiersList()
        done = False

        while not done:

            # Compute in one run the matching set, the best matching classifier and the best matching fitness associated to the previous classifier
            match_set, best_classifier, best_fitness = self.population.form_match_set(state)

            if steps > 0:
                PEPACSClassifiersList.apply_reinforcement_learning(
                    action_set, 
                    last_reward, 
                    best_fitness,
                    self.cfg
                )
                
            # Create action set
            action_set = match_set.form_action_set(best_classifier)
            # Use environment adapter
            iaction = self.cfg.environment_adapter.to_lcs_action(env,best_classifier.action)
            # Do the action
            raw_state, last_reward, terminated, truncated, _info = env.step(iaction)
            done = terminated or truncated
            total_reward += last_reward
            state = self.cfg.environment_adapter.to_genotype(env,raw_state)

            if done:
                # Apply algorithms
                PEPACSClassifiersList.apply_reinforcement_learning(
                    action_set, 
                    last_reward, 
                    0.,
                    self.cfg
                )

            steps += 1

        return TrialMetrics(steps, total_reward)
