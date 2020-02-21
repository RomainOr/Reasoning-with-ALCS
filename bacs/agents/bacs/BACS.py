import logging
from typing import List, Tuple

from bacs import Perception
from bacs.agents.Agent import Agent, TrialMetrics
from bacs.agents.bacs import Classifier, ClassifiersList, Configuration
from bacs.agents.bacs.Condition import Condition
from bacs.agents.bacs.Effect import Effect
from bacs.agents.bacs.components.subsumption_bacs import does_subsume
from bacs.agents.bacs.components.action_selection_bacs import choose_classifier, choose_fittest_classifier

import numpy as np

logger = logging.getLogger(__name__)


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

    def clean_population(self, does_anticipate_change:bool = True):
        compact_pop = []
        for cl in self.population:
            to_keep = True
            for other in self.population:
                if does_subsume(other, cl, self.cfg.theta_exp):
                    to_keep = False
                    break
            if to_keep:
                compact_pop.append(cl)
        self.population = ClassifiersList(*compact_pop)

        if does_anticipate_change:
            pop = [cl for cl in self.population if cl.does_anticipate_change()]
            self.population = ClassifiersList(*pop)

    def _run_trial_explore(self, env, time, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial explore ** ")
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
            match_set = self.population.form_match_set(state)

            if steps > 0:
                # Apply learning in the last action set
                if is_behavioral_sequence:
                    ClassifiersList.apply_alp_behavioral_sequence(
                        self.population,
                        match_set,
                        action_set,
                        prev_state,
                        action,
                        state,
                        time + steps,
                        self.cfg.theta_exp,
                        self.cfg)
                else:
                    ClassifiersList.apply_alp(
                        self.population,
                        match_set,
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
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
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
                        self.cfg.do_subsumption,
                        self.cfg.theta_exp)
            
            action_classifier = choose_classifier(match_set, self.cfg, self.cfg.epsilon)

            is_behavioral_sequence = False
            action = action_classifier.action
            # Use environment adapter and create action set
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            logger.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action)
            # Do the action
            prev_state = state
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            # Enter the if condition only if we have chosen a behavioral classifier
            if action_classifier.behavioral_sequence :
                # Constitute the behavioral learning set
                behavioral_set = match_set.form_behavioral_sequence_set(action_classifier)
                is_behavioral_sequence = True
                # Initialize the message list usefull to decrease quality of classifiers containing looping sequences
                message_list = [state]
                for act in action_classifier.behavioral_sequence:
                    action = act
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(act)
                    logger.debug("\tExecuting action from behavioral sequence: [%d]", act)
                    raw_state, last_reward, done, _ = env.step(iaction)
                    state = self.cfg.environment_adapter.to_genotype(raw_state)
                    if state in message_list:
                        for cl in behavioral_set:    
                            cl.decrease_quality()
                    else:
                        message_list.append(state)
                    if done:
                        break
                    steps += 1
                # Create action set from the learning set
                action_set = behavioral_set

            #Record last activated classifier
            t_2_activated_classifier = t_1_activated_classifier
            t_1_activated_classifier = action_classifier

            if done:
                # Apply algorithms
                if is_behavioral_sequence:
                    ClassifiersList.apply_alp_behavioral_sequence(
                        self.population,
                        ClassifiersList(),
                        action_set,
                        prev_state,
                        action,
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
                    self.cfg.beta,
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
                    self.cfg.do_subsumption,
                    self.cfg.theta_exp)

            steps += 1
        return TrialMetrics(steps, last_reward)

    def _run_trial_exploit(self, env, time=None, current_trial=None) \
            -> TrialMetrics:

        logger.debug("** Running trial exploit **")
        # Initial conditions
        steps = 0
        raw_state = env.reset()
        state = self.cfg.environment_adapter.to_genotype(raw_state)
        
        last_reward = 0
        action_set = ClassifiersList()
        done = False

        while not done:

            match_set = self.population.form_match_set(state)

            if steps > 0:
                ClassifiersList.apply_reinforcement_learning(
                    action_set,
                    last_reward,
                    match_set.get_maximum_fitness(),
                    self.cfg.beta,
                    self.cfg.gamma)

            action_classifier = choose_fittest_classifier(match_set, self.cfg)
            action = action_classifier.action
            # Use environment adapter and create action set
            iaction = self.cfg.environment_adapter.to_lcs_action(action)
            logger.debug("\tExecuting action: [%d]", action)
            action_set = match_set.form_action_set(action)
            # Do the action
            raw_state, last_reward, done, _ = env.step(iaction)
            state = self.cfg.environment_adapter.to_genotype(raw_state)

            # Enter the if condition only if we have chosen a behavioral classifier
            if action_classifier.behavioral_sequence :
                # Constitute the behavioral learning set
                action_set = match_set.form_behavioral_sequence_set(action_classifier)
                for act in action_classifier.behavioral_sequence:
                    # Use environment adapter to execute the action act and perceive its results
                    iaction = self.cfg.environment_adapter.to_lcs_action(act)
                    logger.debug("\tExecuting action from behavioral sequence: [%d]", act)
                    raw_state, last_reward, done, _ = env.step(iaction)
                    state = self.cfg.environment_adapter.to_genotype(raw_state)
                    if done:
                        break
                    steps += 1

            if done:
                # Apply algorithms
                ClassifiersList.apply_reinforcement_learning(
                    action_set, last_reward, 0, self.cfg.beta, self.cfg.gamma)

            steps += 1

        return TrialMetrics(steps, last_reward)
