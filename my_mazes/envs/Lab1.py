from ..envs import AbstractMaze

import numpy as np


class Lab1(AbstractMaze):
    def __init__(self):
        super().__init__(np.matrix([
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
        ]))

    def _insert_animat(self):
        starting_position_x = [1,1,12,12]
        starting_position_y = [1,10,1,10]
        r = np.random.randint(4)
        self.pos_x = starting_position_x[r]
        self.pos_y = starting_position_y[r]