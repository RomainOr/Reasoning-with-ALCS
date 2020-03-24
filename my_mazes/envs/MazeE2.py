"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Anthony J Bagnall and Zhanna V Zatuchna. 2005. On the classification of maze
    problems. In Foundations of Learning Classifier Systems. Springer, 305–316.
"""

from ..envs import AbstractMaze

import numpy as np

class MazeE2(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 9, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, -1, -1, -1, -1, -1, 0, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, -1, -1, 0, 0, 0, -1, -1, 1],
            [1, -1, -1, 0, 9, 0, -1, -1, 1],
            [1, -1, -1, 0, 0, 0, -1, -1, 1],
            [1, -1, -1, -1, -1, -1, -1, -1, 1],
            [1, 0, -1, -1, -1, -1, -1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ])
        )
