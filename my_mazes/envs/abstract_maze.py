"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

import io
import random
import sys
import gym
import networkx as nx
import numpy as np
from gym import spaces, utils

from .. import find_action_by_direction
from .. import ACTION_LOOKUP
from ..maze import Maze, WALL_MAPPING, PATH_MAPPING, ALIASING_MAPPING, REWARD_MAPPING
from ..utils import create_graph

ANIMAT_MARKER = 5


class MazeObservationSpace(gym.Space):
    def __init__(self, n):
        # n is the number of visible neighbour fields, typically 8
        self.n = n
        gym.Space.__init__(self, (self.n,), str)

    def sample(self):
        return tuple(random.choice([str(PATH_MAPPING), str(WALL_MAPPING), str(REWARD_MAPPING)]) for _ in range(self.n))

    def contains(self, x):
        return all(elem in (str(PATH_MAPPING), str(WALL_MAPPING), str(REWARD_MAPPING), str(ANIMAT_MARKER)) for elem in x)

    def to_jsonable(self, sample_n):
        return list(sample_n)

    def from_jsonable(self, sample_n):
        return tuple(sample_n)


class AbstractMaze(gym.Env):
    metadata = {'render.modes': ['human', 'ansi', 'aliasing_human']}

    def __init__(self, matrix, aliasing_matrix):
        self.maze = Maze(matrix,aliasing_matrix)
        self.pos_x = None
        self.pos_y = None
        self.action_space = spaces.Discrete(8)
        self.observation_space = MazeObservationSpace(8)
        self.prob_slippery = 0.0

    def set_prob_slippery(self, prob: float = 0.0):
        self.prob_slippery = prob

    def step(self, action):
        previous_observation = self._observe()
        if random.random() < self.prob_slippery:
            self._take_action(
                random.randint(0, len(ACTION_LOOKUP)-1),
                previous_observation
            )
        else:
            self._take_action(
                action, 
                previous_observation
            )

        observation = self._observe()
        reward = self._get_reward()
        episode_over = self._is_over()

        return observation, reward, episode_over, {}

    def reset(self):
        self._insert_animat()
        return self._observe()

    def render(self, mode='aliasing_human'):
        if mode == 'aliasing_human':
            self._render_to_file(sys.stdout, aliasing_mode=True)
        elif mode == 'human':
            self._render_to_file(sys.stdout, aliasing_mode=False)
        elif mode == 'ansi':
            output = io.StringIO()
            self._render_to_file(output, aliasing_mode=False)
            return output.getvalue()
        else:
            super(AbstractMaze, self).render(mode=mode)

    def get_all_possible_transitions(self):
        transitions = []
        g = create_graph(self)
        path_nodes = (node for node, data
            in g.nodes(data=True) if data['type'] == 'path')
        for node in path_nodes:
            for neighbour in nx.all_neighbors(g, node):
                direction = Maze.distinguish_direction(node, neighbour)
                action = find_action_by_direction(direction)
                transitions.append((node, action, neighbour))
        return transitions

    def get_all_aliased_states(self):
        all_aliased_states = []
        for y in range(0, self.maze.max_y):
            for x in range(0, self.maze.max_x):
                if self.maze.is_aliased(x,y):
                    all_aliased_states.append(self.maze.perception(x, y))
        return list(dict.fromkeys(all_aliased_states))

    def get_all_non_aliased_states(self):
        all_non_aliased_states = []
        for y in range(0, self.maze.max_y):
            for x in range(0, self.maze.max_x):
                if self.maze.is_path_in_aliasing_matrix(x,y):
                    all_non_aliased_states.append(self.maze.perception(x, y))
        return all_non_aliased_states

    def _observe(self):
        return self.maze.perception(self.pos_x, self.pos_y)

    def _get_reward(self):
        if self.maze.is_reward(self.pos_x, self.pos_y):
            return 1000

        return 0

    def _is_over(self):
        return self.maze.is_reward(self.pos_x, self.pos_y)

    def _take_action(self, action, observation):
        """Executes the action inside the maze"""
        animat_moved = False
        action_type = ACTION_LOOKUP[action]

        if action_type == "N" and not self.is_wall(observation[0]):
            self.pos_y -= 1
            animat_moved = True

        if action_type == 'NE' and not self.is_wall(observation[1]):
            self.pos_x += 1
            self.pos_y -= 1
            animat_moved = True

        if action_type == "E" and not self.is_wall(observation[2]):
            self.pos_x += 1
            animat_moved = True

        if action_type == 'SE' and not self.is_wall(observation[3]):
            self.pos_x += 1
            self.pos_y += 1
            animat_moved = True

        if action_type == "S" and not self.is_wall(observation[4]):
            self.pos_y += 1
            animat_moved = True

        if action_type == 'SW' and not self.is_wall(observation[5]):
            self.pos_x -= 1
            self.pos_y += 1
            animat_moved = True

        if action_type == "W" and not self.is_wall(observation[6]):
            self.pos_x -= 1
            animat_moved = True

        if action_type == 'NW' and not self.is_wall(observation[7]):
            self.pos_x -= 1
            self.pos_y -= 1
            animat_moved = True

        return animat_moved

    def _insert_animat(self):
        possible_coords = self.maze.get_possible_insertion_coordinates()
        starting_position = random.choice(possible_coords)
        self.pos_x = starting_position[0]
        self.pos_y = starting_position[1]

    def _render_to_file(self, outfile, aliasing_mode):
        outfile.write("\n")
        if aliasing_mode:
            situation = np.copy(self.maze.aliasing_matrix)
        else:
            situation = np.copy(self.maze.matrix)
        situation[self.pos_y, self.pos_x] = ANIMAT_MARKER
        for row in situation:
            outfile.write(" ".join(self._render_element(el) for el in row))
            outfile.write("\n")

    @staticmethod
    def is_wall(perception):
        return perception == str(WALL_MAPPING)

    @staticmethod
    def _render_element(el):
        if el == WALL_MAPPING:
            return utils.colorize('■', 'gray')
        elif el == PATH_MAPPING:
            return utils.colorize('□', 'white')
        elif el == REWARD_MAPPING:
            return utils.colorize('$', 'yellow')
        elif el == ANIMAT_MARKER:
            return utils.colorize('A', 'red')
        elif el == ALIASING_MAPPING:
            return utils.colorize('■', 'cyan')
        else:
            return utils.colorize(el, 'cyan')
