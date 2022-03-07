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
from ..maze import Maze, WALL_MAPPING, PATH_MAPPING, ALIASING_MAPPING, REWARD_MAPPING, OBSTACLE_MAPPING
from ..utils import create_graph

ANIMAT_MARKER = 5


class MazeObservationSpace(gym.Space):
    def __init__(self, n):
        # n is the number of visible neighbour fields, typically 8
        self.n = n
        gym.Space.__init__(self, (self.n,), str)

    def sample(self):
        return tuple(random.choice([str(PATH_MAPPING), str(WALL_MAPPING), str(REWARD_MAPPING), str(OBSTACLE_MAPPING)]) for _ in range(self.n))

    def contains(self, x):
        return all(elem in (str(PATH_MAPPING), str(WALL_MAPPING), str(REWARD_MAPPING), str(ANIMAT_MARKER), str(OBSTACLE_MAPPING)) for elem in x)

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
        self.random_attribute_length = 0
        self.reward = 1000
        self.obstacle_reward = -1000

    def set_obstacle_reward(self, r: float = 0.0):
        self.obstacle_reward = r

    def set_reward(self, r: float = 0.0):
        self.reward = r

    def set_prob_slippery(self, prob: float = 0.0):
        self.prob_slippery = prob

    def set_random_attribute_length(self, l: int = 0):
        self.random_attribute_length = l

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

    def get_theoritical_probabilities(self):
        # get all transitions
        transitions = []
        for node, action, neighbour in self.get_all_possible_transitions():
            transitions.append((
                "".join(self.maze.perception(*node)), 
                action, 
                "".join(self.maze.perception(*neighbour))
            ))
        # building the data structure by computing all reachable states depending on the action for every position
        result = {}
        for node, action, neighbour in transitions:
            if node in result:
                if action in result[node]:
                    result[node][action]["reachable_states"].append(neighbour)
                else:
                    result[node][action] = {"reachable_states":[neighbour]}
            else :
                result[node] = {action:{"reachable_states":[neighbour]}}
        # completing the data structure with transitions that do not lead to a change in the environment
        for key, value in result.items():
            for action in ACTION_LOOKUP:
                if action not in value:
                    value[action] = {"reachable_states":[key]}
        # adding to the structure the theoritical probabilities to perceive each item
        number_of_actions = len(ACTION_LOOKUP)
        for _start_state, action_and_reachable_states in result.items():
            for key in action_and_reachable_states:
                action_and_reachable_states[key]["probabilities"] = {}
                theoritical_probabilities = action_and_reachable_states[key]["probabilities"]
                reachable_states = action_and_reachable_states[key]["reachable_states"]
                # 1° set up the probabilities depending on slippery if the action done is the expected one
                for a in ACTION_LOOKUP:
                    theoritical_probabilities[a] = {}
                    for i in range(len(reachable_states)):
                        if int(reachable_states[i][a]) in theoritical_probabilities[a]:
                            theoritical_probabilities[a][int(reachable_states[i][a])] += 1. - (number_of_actions -1 ) * self.prob_slippery / number_of_actions
                        else:
                            theoritical_probabilities[a][int(reachable_states[i][a])] = 1. - (number_of_actions -1 ) * self.prob_slippery / number_of_actions
                # 2° taking in consideration the aliasing states due to the perceptual aliasing issue 
                for _, prob in action_and_reachable_states[key]["probabilities"].items():
                    for symbol in prob:
                        prob[symbol] /= len(reachable_states)
                # 3° update the probabilities depending on slippery if the action done is not the expected one
            slip_rate_by_action = self.prob_slippery / number_of_actions #(uniform distribution)
            for selected_action in action_and_reachable_states:
                theoritical_probabilities = action_and_reachable_states[selected_action]["probabilities"]
                for slippery_action in ACTION_LOOKUP:
                    if selected_action != slippery_action:
                        slippery_reachable_states = action_and_reachable_states[slippery_action]["reachable_states"]
                        for a in ACTION_LOOKUP:
                            for i in range(len(slippery_reachable_states)):
                                if int(slippery_reachable_states[i][a]) in theoritical_probabilities[a]:
                                    theoritical_probabilities[a][int(slippery_reachable_states[i][a])] += slip_rate_by_action / len(slippery_reachable_states)
                                else:
                                    theoritical_probabilities[a][int(slippery_reachable_states[i][a])] = slip_rate_by_action / len(slippery_reachable_states)
        return result

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
        n, ne, e, se, s, sw, w, nw =  self.maze.perception(self.pos_x, self.pos_y)
        if self.random_attribute_length == 1:
            return n, ne, e, se, s, sw, w, nw, str(random.randint(0, 1))
        else:
            return n, ne, e, se, s, sw, w, nw

    def _get_reward(self):
        if self.maze.is_reward(self.pos_x, self.pos_y):
            return self.reward
        if self.maze.is_obstacle(self.pos_x, self.pos_y):
            return self.obstacle_reward

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
        elif el == OBSTACLE_MAPPING:
            return utils.colorize('■', 'magenta')
        elif el == ANIMAT_MARKER:
            return utils.colorize('A', 'red')
        elif el == ALIASING_MAPPING:
            return utils.colorize('■', 'cyan')
        else:
            return utils.colorize(el, 'cyan')
