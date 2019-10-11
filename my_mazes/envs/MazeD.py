from ..envs import AbstractMaze

import numpy as np


class MazeD(AbstractMaze):
    def __init__(self):
        super().__init__(np.matrix([
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 1, 0, 1],
            [1, 0, 0, 1, 1, 9, 1, 1],
            [1, 0, 0, 0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1]
        ]))

    def _insert_animat(self):
        self.pos_x = 1
        self.pos_y = 1