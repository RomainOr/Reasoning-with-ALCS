from ..envs import AbstractMaze

import numpy as np


class Woods100(AbstractMaze):
    def __init__(self):
        super().__init__(np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 9, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ]))
