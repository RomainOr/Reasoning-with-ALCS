"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Tomohiro Hayashida, Ichiro Nishizaki, and Ryosuke Sakato. 2014. Aliased states
    discerning in POMDPs and improved anticipatory classifier system. Procedia
    Computer Science 35 (2014), 34–43.
"""

from ..envs import AbstractMaze

import numpy as np

# TODO : Have to build aliasing matrix

class Lab1(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 9, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 0, 0, 1, 9, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
            [1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            ])
        )

    def _insert_animat(self):
        starting_position_x = [1,1,12,12]
        starting_position_y = [1,10,1,10]
        r = np.random.randint(4)
        self.pos_x = starting_position_x[r]
        self.pos_y = starting_position_y[r]