"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Martin V Butz, David E Goldberg, and Wolfgang Stolzmann. 2000. 
    Probability-enhanced predictions in the anticipatory classifier system.
    In International Workshop on Learning Classifier Systems. Springer, 37–51.
"""

from ..envs import MazeGymEnv

import numpy as np

class Maze4(MazeGymEnv):
    def __init__(self, slippery_prob=0., render_mode='aliasing_human'):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 9, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 0, 9, 1],
            [1, 1, 0, 0, 1, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
            ]),
            slippery_prob=slippery_prob,
            render_mode=render_mode
        )
