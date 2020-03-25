"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Callable

from bacs.agents import EnvironmentAdapter


class Configuration:

    def __init__(self,
                 classifier_length: int,
                 number_of_possible_actions: int,
                 classifier_wildcard='#',
                 environment_adapter=EnvironmentAdapter,
                 user_metrics_collector_fcn: Callable = None,
                 metrics_trial_frequency: int = 5,
                 do_pee: bool = True,
                 do_ga: bool=False,
                 beta: float=0.05,
                 gamma: float=0.95,
                 theta_i: float=0.1,
                 theta_r: float=0.9,
                 epsilon: float=0.5,
                 u_max: int=100000,
                 theta_exp: int=20,
                 theta_ga: int=100,
                 theta_as: int=20,
                 mu: float=0.3,
                 chi: float=0.8,
                 bs_max: int=1) -> None:
        """
        Creates the configuration object used during training the BACS agent.

        classifier_length
            length of the condition and effect strings
        number_of_possible_actions
            number of possible actions to be executed
        classifier_wildcard

        environment_adapter
            EnvironmentAdapter class BACS needs to use
            to interact with the environment
        do_ga
            switch *Genetic Generalization* module
        beta

        gamma

        theta_i
            inadequacy threshold
        theta_r

        epsilon

        u_max

        theta_exp

        theta_ga

        theta_as

        mu
            GA mutation probability
        chi
            GA crossover probability
        bs_max
            maximal length of behavioral sequence
        """
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = classifier_wildcard
        self.environment_adapter = environment_adapter
        self.metrics_trial_frequency = metrics_trial_frequency
        self.user_metrics_collector_fcn = user_metrics_collector_fcn
        self.do_pee = do_pee
        self.do_ga = do_ga
        self.theta_exp = theta_exp
        self.beta = beta
        self.gamma = gamma
        self.theta_i = theta_i
        self.theta_r = theta_r
        self.epsilon = epsilon
        self.u_max = u_max
        self.theta_ga = theta_ga
        self.theta_as = theta_as
        self.mu = mu
        self.chi = chi
        self.bs_max = bs_max


    def __str__(self):
        return "BACSConfiguration:" \
            "\n\t- Classifier length: [{}]" \
            "\n\t- Number of possible actions: [{}]" \
            "\n\t- Classifier wildcard: [{}]" \
            "\n\t- Environment adapter function: [{}]" \
            "\n\t- Do Pee: [{}]" \
            "\n\t- Do GA: [{}]" \
            "\n\t- Beta: [{}]" \
            "\n\t- ..." \
            "\n\t- epsilon: [{}]" \
            "\n\t- bs_max: [{}]" \
        .format(
            self.classifier_length,
            self.number_of_possible_actions,
            self.classifier_wildcard,
            self.environment_adapter,
            self.do_pee,
            self.do_ga,
            self.beta,
            self.epsilon,
            self.bs_max
        )
