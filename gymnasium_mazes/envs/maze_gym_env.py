"""
    This Source Code Form is subject to the terms of the Mozilla Public
    License, v. 2.0. If a copy of the MPL was not distributed with this
    file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""

from enum import Enum

import io
import sys
import gymnasium as gym
import networkx as nx
import numpy as np

OBSERVATION_MAPPING = {
    'ALIASING'  : -1,
    'PATH'      : 0,
    'WALL'      : 1,
    'ANIMAT'    : 5,
    'EXIT'      : 9
}

class Actions(Enum):
    NORTH = 0
    NORTHEAST = 1
    EAST = 2
    SOUTHEAST = 3
    SOUTH = 4
    SOUTHWEST = 5
    WEST = 6
    NORTHWEST = 7

class MazeObservationSpace(gym.Space):
    def __init__(self, n):
        # n is the number of visible neighbour fields, typically 8
        self.n = n
        gym.Space.__init__(self, (self.n,), str)

    def sample(self):
        return tuple(
            self.np_random.choice(OBSERVATION_MAPPING.values()) for _ in range(self.n))

    def contains(self, x):
        return all(elem in (OBSERVATION_MAPPING.values()) for elem in x)


class MazeGymEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi', 'aliasing_human'], "render_fps": 1}

    def __init__(self, matrix, aliasing_matrix, render_mode='aliasing_human'):
        self.maze = matrix
        self.aliased_maze_to_plot = aliasing_matrix
        self.max_x = self.maze.shape[1]
        self.max_y = self.maze.shape[0]
        self._agent_location = np.array([-1, -1], dtype=int)
        self.action_space = gym.spaces.Discrete(len(Actions))
        self._action_to_direction = {
            Actions.NORTH.value: np.array([0, -1]),
            Actions.NORTHEAST.value: np.array([1, -1]),
            Actions.EAST.value: np.array([1, 0]),
            Actions.SOUTHEAST.value: np.array([1, 1]),
            Actions.SOUTH.value: np.array([0, 1]),
            Actions.SOUTHWEST.value: np.array([-1, 1]),
            Actions.WEST.value: np.array([-1, 0]),
            Actions.NORTHWEST.value: np.array([-1, -1])
        }
        self.observation_space = MazeObservationSpace(8)
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._insert_animat()
        return self._get_obs(), self._get_info()

    def step(self, action):
        previous_observation = self._get_obs()
        if previous_observation[action] != str(OBSERVATION_MAPPING['WALL']):
            self._agent_location += self._action_to_direction[action]
        observation = self._get_obs()
        reward = self._get_reward()
        terminated = self._is_terminated()
        info = self._get_info()
        truncated = False
        return observation, reward, terminated, truncated, info

    def build_perception_from_location(self, pos_x, pos_y):
        if not (0 <= pos_x < self.max_x):
            raise ValueError('X position not within allowed range')
        if not (0 <= pos_y < self.max_y):
            raise ValueError('Y position not within allowed range')
        # Position N
        if pos_y == 0:
            n = None
        else:
            n = str(self.maze[pos_y - 1, pos_x])
        # Position NE
        if pos_x == self.max_x - 1 or pos_y == 0:
            ne = None
        else:
            ne = str(self.maze[pos_y - 1, pos_x + 1])
        # Position E
        if pos_x == self.max_x - 1:
            e = None
        else:
            e = str(self.maze[pos_y, pos_x + 1])
        # Position SE
        if pos_x == self.max_x - 1 or pos_y == self.max_y - 1:
            se = None
        else:
            se = str(self.maze[pos_y + 1, pos_x + 1])
        # Position S
        if pos_y == (self.max_y - 1):
            s = None
        else:
            s = str(self.maze[pos_y + 1, pos_x])
        # Position SW
        if pos_x == 0 or pos_y == self.max_y - 1:
            sw = None
        else:
            sw = str(self.maze[pos_y + 1, pos_x - 1])
        # Position W
        if pos_x == 0:
            w = None
        else:
            w = str(self.maze[pos_y, pos_x - 1])
        # Position NW
        if pos_x == 0 or pos_y == 0:
            nw = None
        else:
            nw = str(self.maze[pos_y - 1, pos_x - 1])
        return n, ne, e, se, s, sw, w, nw

    def _get_obs(self):
        return self.build_perception_from_location(self._agent_location[0], self._agent_location[1])
        
    def _get_info(self):
        return {}

    def _is_exit(self, pos_x, pos_y):
        return self.maze[pos_y, pos_x] == OBSERVATION_MAPPING['EXIT']

    def _get_reward(self):
        if self._is_exit(self._agent_location[0], self._agent_location[1]):
            return 1000
        return 0

    def _is_terminated(self):
        return self._is_exit(self._agent_location[0], self._agent_location[1])

    def _is_path(self, pos_x, pos_y):
        return self.maze[pos_y, pos_x] == OBSERVATION_MAPPING['PATH']

    def _insert_animat(self):
        possible_coords = []
        for x in range(0, self.max_x):
            for y in range(0, self.max_y):
                if self._is_path(x, y):
                    possible_coords.append((x, y))
        starting_position = self.np_random.choice(possible_coords)
        self._agent_location = np.array([starting_position[0], starting_position[1]], dtype=int)

    def render(self):
        if self.render_mode == 'aliasing_human':
            return self._render_to_file(sys.stdout, aliasing_mode=True)
        elif self.render_mode == 'human':
            return self._render_to_file(sys.stdout, aliasing_mode=False)
        elif self.render_mode == 'ansi':
            output = io.StringIO()
            self._render_to_file(output, aliasing_mode=False)
            return output.getvalue()

    def _render_to_file(self, outfile, aliasing_mode):
        outfile.write("\n")
        if aliasing_mode:
            situation = np.copy(self.aliased_maze_to_plot)
        else:
            situation = np.copy(self.maze)
        situation[self._agent_location[1], self._agent_location[0]] = OBSERVATION_MAPPING['ANIMAT']
        for row in situation:
            outfile.write(" ".join(self._render_element(el) for el in row))
            outfile.write("\n")

    def _render_element(self, el):
        if el == OBSERVATION_MAPPING['WALL']:
            return gym.utils.colorize('■', 'gray')
        elif el == OBSERVATION_MAPPING['PATH']:
            return gym.utils.colorize('□', 'white')
        elif el == OBSERVATION_MAPPING['EXIT']:
            return gym.utils.colorize('$', 'yellow')
        elif el == OBSERVATION_MAPPING['ANIMAT']:
            return gym.utils.colorize('A', 'red')
        elif el == OBSERVATION_MAPPING['ALIASING']:
            return gym.utils.colorize('■', 'cyan')
        else:
            return gym.utils.colorize(el, 'black')

    def _get_possible_neighbour_cords(self, pos_x, pos_y) -> tuple:
        """
        Returns a tuple with coordinates for
        N, NE, E, SE, S, SW, W, NW neighbouring cells.
        """
        n = (pos_x, pos_y - 1)
        ne = (pos_x + 1, pos_y - 1)
        e = (pos_x + 1, pos_y)
        se = (pos_x + 1, pos_y + 1)
        s = (pos_x, pos_y + 1)
        sw = (pos_x - 1, pos_y + 1)
        w = (pos_x - 1, pos_y)
        nw = (pos_x - 1, pos_y - 1)
        return n, ne, e, se, s, sw, w, nw
    
    def _create_graph(self):
        # Create uni-directed graph
        g = nx.Graph()
        # Add nodes
        for x in range(0, self.max_x):
            for y in range(0, self.max_y):
                if self._is_path(x, y):
                    g.add_node((x, y), type='path')
                if self._is_exit(x, y):
                    g.add_node((x, y), type='exit')
        # Add edges
        path_nodes = [cords for cords, attribs
            in g.nodes(data=True) if attribs['type'] == 'path']
        for n in path_nodes:
            neighbour_cells = self._get_possible_neighbour_cords(*n)
            allowed_cells = [c for c in neighbour_cells
                if self._is_path(*c) or self._is_exit(*c)]
            edges = [(n, dest) for dest in allowed_cells]
            g.add_edges_from(edges)
        return g

    def _distinguish_direction(self, start, end):
        direction = ''
        if end[1] + 1 == start[1]:
            direction += 'N'
        if end[1] - 1 == start[1]:
            direction += 'S'
        if end[0] + 1 == start[0]:
            direction += 'W'
        if end[0] - 1 == start[0]:
            direction += 'E'
        return direction

    def get_all_aliased_states(self):
        all_aliased_states = []
        for x in range(0, self.max_x):
            for y in range(0, self.max_y):
                if self.aliased_maze_to_plot[y, x] == OBSERVATION_MAPPING['ALIASING']:
                    all_aliased_states.append(self.build_perception_from_location(x, y))
        return list(dict.fromkeys(all_aliased_states))

    def get_all_non_aliased_states(self):
        all_non_aliased_states = []
        for x in range(0, self.max_x):
            for y in range(0, self.max_y):
                if self.aliased_maze_to_plot[y, x] == OBSERVATION_MAPPING['PATH']:
                    all_non_aliased_states.append(self.build_perception_from_location(x, y))
        return all_non_aliased_states

    def get_all_possible_transitions(self):
        DIRECTION_LOOKUP = {
            'N' : Actions.NORTH.value,
            'NE': Actions.NORTHEAST.value,
            'E' : Actions.EAST.value,
            'SE': Actions.SOUTHEAST.value,
            'S' : Actions.SOUTH.value,
            'SW': Actions.SOUTHWEST.value,
            'W' : Actions.WEST.value,
            'NW': Actions.NORTHWEST.value
        }
        transitions = []
        g = self._create_graph()
        path_nodes = (node for node, data
            in g.nodes(data=True) if data['type'] == 'path')
        for node in path_nodes:
            for neighbour in nx.all_neighbors(g, node):
                direction = self._distinguish_direction(node, neighbour)
                transitions.append((node, DIRECTION_LOOKUP[direction], neighbour))
        return transitions

    def get_theoritical_probabilities(self, slippery_prob = 0.):
        # get all transitions
        transitions = []
        for node, action, neighbour in self.get_all_possible_transitions():
            transitions.append((
                "".join(self.build_perception_from_location(*node)), 
                action, 
                "".join(self.build_perception_from_location(*neighbour))
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
            for action in Actions:
                if action.value not in value:
                    value[action.value] = {"reachable_states":[key]}
        # adding to the structure the theoritical probabilities to perceive each item
        number_of_actions = len(Actions)
        for _start_state, action_and_reachable_states in result.items():
            for key in action_and_reachable_states:
                action_and_reachable_states[key]["probabilities"] = {}
                theoritical_probabilities = action_and_reachable_states[key]["probabilities"]
                reachable_states = action_and_reachable_states[key]["reachable_states"]
                # 1° set up the probabilities depending on slippery if the action done is the expected one
                for a in Actions:
                    theoritical_probabilities[a.value] = {}
                    for i in range(len(reachable_states)):
                        if int(reachable_states[i][a.value]) in theoritical_probabilities[a.value]:
                            theoritical_probabilities[a.value][int(reachable_states[i][a.value])] += 1. - (number_of_actions -1 ) * slippery_prob / number_of_actions
                        else:
                            theoritical_probabilities[a.value][int(reachable_states[i][a.value])] = 1. - (number_of_actions -1 ) * slippery_prob / number_of_actions
                # 2° taking in consideration the aliasing states due to the perceptual aliasing issue 
                for _, prob in action_and_reachable_states[key]["probabilities"].items():
                    for symbol in prob:
                        prob[symbol] /= len(reachable_states)
                # 3° update the probabilities depending on slippery if the action done is not the expected one
            slip_rate_by_action = slippery_prob / number_of_actions #(uniform distribution)
            for selected_action in action_and_reachable_states:
                theoritical_probabilities = action_and_reachable_states[selected_action]["probabilities"]
                for slippery_action in Actions:
                    if selected_action != slippery_action.value:
                        slippery_reachable_states = action_and_reachable_states[slippery_action.value]["reachable_states"]
                        for a in Actions:
                            for i in range(len(slippery_reachable_states)):
                                if int(slippery_reachable_states[i][a.value]) in theoritical_probabilities[a.value]:
                                    theoritical_probabilities[a.value][int(slippery_reachable_states[i][a.value])] += slip_rate_by_action / len(slippery_reachable_states)
                                else:
                                    theoritical_probabilities[a.value][int(slippery_reachable_states[i][a.value])] = slip_rate_by_action / len(slippery_reachable_states)
        return result
