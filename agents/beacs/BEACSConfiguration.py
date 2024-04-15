"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from typing import Callable

from agents.common.EnvironmentAdapter import EnvironmentAdapter
from agents.common.BaseConfiguration import BaseConfiguration


class BEACSConfiguration(BaseConfiguration):

    def __init__(self,
            classifier_length: int,
            number_of_possible_actions: int,
            classifier_wildcard='#',
            environment_adapter=EnvironmentAdapter(),
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
            theta_bseq: int=1000,
            bs_max: int=0,
            do_ep: bool = True,
        ) -> None:
        """
        Creates the configuration object used during training the beacs agent.
        """
        super().__init__(
            classifier_length=classifier_length,
            number_of_possible_actions=number_of_possible_actions,
            classifier_wildcard=classifier_wildcard,
            environment_adapter=environment_adapter,
            user_metrics_collector_fcn=user_metrics_collector_fcn,
            metrics_trial_frequency=metrics_trial_frequency,
            epsilon=epsilon,
            seed=seed,
            beta_alp=beta_alp,
            theta_i=theta_i,
            theta_r=theta_r,
            theta_exp=theta_exp,
            u_max=u_max,
            beta_rl=beta_rl,
            gamma=gamma,
            theta_ga=theta_ga,
            theta_as=theta_as,
            mu=mu,
            chi=chi
        )
        self.theta_bseq = theta_bseq
        self.bs_max = bs_max
        self.do_ep = do_ep


    def __str__(self) -> str:
        return "BEACS additional parameters:" \
            "\n\t- Theta_bseq: [{}]" \
            "\n\t- bs_max: [{}]" \
            "\n\t- Do enhanced prediction: [{}]" \
        .format(
            self.theta_bseq,
            self.bs_max,
            self.do_ep
        ) + super().__str__()
