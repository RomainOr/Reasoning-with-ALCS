"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Uwano, Fumito, and Will Browne. "Hierarchical Frames-of-References in Learning Classifier Systems." 
    Proceedings of the Companion Conference on Genetic and Evolutionary Computation. 2023.
"""

from ..envs import AbstractMaze

import numpy as np

class MazeF8(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 9, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            np.matrix([
            [1, 1, 1,  1,  1,  1,  1,  1, 1, 1, 1],
            [1, 0, 0, -1, -1, -1, -1, -1, 0, 9, 1],
            [1, 0, 1,  1,  1,  1,  1,  1, 1, 1, 1],
            [1, 0, 0, -1, -1, -1, -1, -1, 0, 1, 1],
            [1, 0, 1,  1,  1,  1,  1,  1, 1, 1, 1],
            [1, 1, 1,  1,  1,  1,  1,  1, 1, 1, 1],
            ])
        )
