"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Sachiyo Arai and Katia Sycara. 2001. Credit assignment method for learning
    effective stochastic policies in uncertain domains. In Proceedings of the 3rd Annual
    Conference on Genetic and Evolutionary Computation. 815–822.
"""

from ..envs import AbstractMaze

import numpy as np

class MazeB(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 9, 1],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [1, 0, 0, 1, -1, 1, 9, 1],
            [1, 1, -1, 1, -1, 0, 0, 1],
            [1, 1, -1, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 1, 0, 1],
            [1, 0, -1, 0, 0, -1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
            ])
        )