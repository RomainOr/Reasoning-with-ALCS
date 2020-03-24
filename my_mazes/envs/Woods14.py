"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Martin V Butz, David E Goldberg, and Wolfgang Stolzmann. 2002. The anticipatory
    classifier system and genetic generalization. Natural Computing 1, 4 (2002),
    427–467.
"""

from ..envs import AbstractMaze

import numpy as np

class Woods14(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 9, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1],
            [1, 9, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ])
        )
