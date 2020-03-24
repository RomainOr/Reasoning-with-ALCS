"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Zhaoxiang Zang, Dehua Li, and Junying Wang. 2015. Learning classifier systems
    with memory condition to solve non-Markov problems. Soft Computing 19, 6
    (2015), 1679–1699.
"""

from ..envs import AbstractMaze

import numpy as np

class Woods101demi(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 9, 1, 0, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 0, 1, 0, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 0, 1, 9, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1],
            [1, -1, 1, 9, 1, -1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, -1, 1, -1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, 1, -1, 1, -1, 1, 1],
            [1, 0, 1, 0, 1, 0, 1],
            [1, -1, 1, 9, 1, -1, 1],
            [1, 1, 1, 1, 1, 1, 1],
            ])
        )
