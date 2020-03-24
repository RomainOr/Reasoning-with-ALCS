"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Marc Métivier and Claude Lattaud. 2002. Anticipatory classifier system using
    behavioral sequences in non-markov environments. In International Workshop
    on Learning Classifier Systems. Springer, 143–162.
"""

from ..envs import AbstractMaze

import numpy as np

class Woods100(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 0, 0, 9, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            np.matrix([                
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, -1, 0, 9, 0, -1, 0, 1],
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
            ])
        )
