"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Martin V Butz. 2001. Biasing exploration in an anticipatory learning classifier
    system. In International Workshop on Learning Classifier Systems. Springer, 3–22.
"""

from ..envs import AbstractMaze

import numpy as np

class Maze5(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 9, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 9, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 1, 1, 0, 0, 1],
            [1, 0, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 1, 0, 0, 1, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ])
        )
