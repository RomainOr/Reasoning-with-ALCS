"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

"""
    Kazuteru Miyazaki and Shigenobu Kobayashi. 1999. Proposal for an algorithm to
    improve a rational policy in POMDPs. In IEEE SMC’99 Conference Proceedings.
    1999 IEEE International Conference on Systems, Man, and Cybernetics (Cat. No.
    99CH37028), Vol. 5. IEEE, 492–497.
"""

from ..envs import AbstractMaze

import numpy as np

class MiyazakiB(AbstractMaze):
    def __init__(self):
        super().__init__(
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1],
            [1, 1, 0, 0, 0, 1, 9, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 0, 0, 1, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            ]),
            np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 1, 0, 1, 1, 1],
            [1, 0, -1, 0, 0, 1, 1, 1],
            [1, 1, -1, -1, 0, 1, 9, 1],
            [1, 0, 0, 0, -1, 0, 0, 1],
            [1, 0, 0, 1, -1, -1, 0, 1],
            [1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
            ])
        )
