"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Callable

from agents.common import EnvironmentAdapter


class BaseConfiguration():

    def __init__(self,
            classifier_length: int,
            number_of_possible_actions: int,
            classifier_wildcard='#',
            environment_adapter=EnvironmentAdapter,
            user_metrics_collector_fcn: Callable = None,
            metrics_trial_frequency: int = 5,
            epsilon: float=0.5,
            seed:int = None,
            beta_alp: float=0.05,
            theta_i: float=0.1,
            theta_r: float=0.9,
            theta_exp: int=20,
            u_max: int=100000,
            beta_rl: float=0.05,
            gamma: float=0.95,
            theta_ga: int=100,
            theta_as: int=20,
            mu: float=0.3,
            chi: float=0.8,
        ) -> None:
        """
        Creates the configuration object used during training the agent.
        """
        self.classifier_length = classifier_length
        self.number_of_possible_actions = number_of_possible_actions
        self.classifier_wildcard = classifier_wildcard
        self.environment_adapter = environment_adapter
        self.user_metrics_collector_fcn = user_metrics_collector_fcn
        self.metrics_trial_frequency = metrics_trial_frequency
        self.epsilon = epsilon
        self.seed = seed
        self.beta_alp = beta_alp
        self.theta_i = theta_i
        self.theta_r = theta_r
        self.theta_exp = theta_exp
        self.u_max = u_max
        self.beta_rl = beta_rl
        self.gamma = gamma
        self.theta_ga = theta_ga
        self.theta_as = theta_as
        self.mu = mu
        self.chi = chi


    def __str__(self):
        return "BaseConfiguration:" \
            "\n\t- Classifier length: [{}]" \
            "\n\t- Number of possible actions: [{}]" \
            "\n\t- Classifier wildcard: [{}]" \
            "\n\t- Environment adapter function: [{}]" \
            "\n\t- User collector metric function: [{}]" \
            "\n\t- Metric trial frequency: [{}]" \
            "\n\t- epsilon: [{}]" \
            "\n\t- seed: [{}]" \
            "\nALP Configuration:" \
            "\n\t- Beta_ALP: [{}]" \
            "\n\t- Theta_i: [{}]" \
            "\n\t- Theta_r: [{}]" \
            "\n\t- Theta_exp: [{}]" \
            "\n\t- u_max: [{}]" \
            "\nRL Configuration:" \
            "\n\t- Beta_RL: [{}]" \
            "\n\t- Gamma: [{}]" \
            "\nGA Configuration:" \
            "\n\t- Theta_ga: [{}]" \
            "\n\t- Theta_as: [{}]" \
            "\n\t- Mu: [{}]" \
            "\n\t- Chi: [{}]" \
        .format(
            self.classifier_length,
            self.number_of_possible_actions,
            self.classifier_wildcard,
            self.environment_adapter,
            self.user_metrics_collector_fcn,
            self.metrics_trial_frequency,
            self.epsilon,
            self.seed,
            self.beta_alp,
            self.theta_i,
            self.theta_r,
            self.theta_exp,
            self.u_max,
            self.beta_rl,
            self.gamma,
            self.theta_ga,
            self.theta_as,
            self.mu,
            self.chi
        )
