"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    John Loch and Satinder P Singh. 1998. Using Eligibility Traces to Find the Best
    Memoryless Policy in Partially Observable Markov Decision Processes.. In ICML.
    323–331.
"""

from ..envs import AbstractMaze

import numpy as np

class Littman89(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 9, 1],
            [1, 1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, -1, 0, -1, 0, 1, 1],
            [1, -1, 0, 1, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 1, 0, 1, 1],
            [1, -1, 0, 1, 0, 1, 0, 9, 1],
            [1, 1, 0, -1, 0, -1, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1]
            ])
        )
