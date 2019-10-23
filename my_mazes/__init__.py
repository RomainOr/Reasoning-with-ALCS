from gym.envs.registration import register

# noinspection PyUnresolvedReferences
from .maze import Maze, PATH_MAPPING, WALL_MAPPING, REWARD_MAPPING

ACTION_LOOKUP = {
    0: 'N',
    1: 'NE',
    2: 'E',
    3: 'SE',
    4: 'S',
    5: 'SW',
    6: 'W',
    7: 'NW'
}


def find_action_by_direction(direction):
    for key, val in ACTION_LOOKUP.items():
        if val == direction:
            return key

register(
    id='Cassandra4x4-v0',
    entry_point='my_mazes.envs:Cassandra4x4',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Lab1-v0',
    entry_point='my_mazes.envs:Lab1',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Littman57-v0',
    entry_point='my_mazes.envs:Littman57',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Littman89-v0',
    entry_point='my_mazes.envs:Littman89',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeA-v0',
    entry_point='my_mazes.envs:MazeA',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeB-v0',
    entry_point='my_mazes.envs:MazeB',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeD-v0',
    entry_point='my_mazes.envs:MazeD',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeF4-v0',
    entry_point='my_mazes.envs:MazeF4',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Maze4-v0',
    entry_point='my_mazes.envs:Maze4',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Maze5-v0',
    entry_point='my_mazes.envs:Maze5',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Maze6-v0',
    entry_point='my_mazes.envs:Maze6',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Maze7-v0',
    entry_point='my_mazes.envs:Maze7',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Maze10-v0',
    entry_point='my_mazes.envs:Maze10',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeE1-v0',
    entry_point='my_mazes.envs:MazeE1',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeE2-v0',
    entry_point='my_mazes.envs:MazeE2',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeE3-v0',
    entry_point='my_mazes.envs:MazeE3',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeF1-v0',
    entry_point='my_mazes.envs:MazeF1',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeF2-v0',
    entry_point='my_mazes.envs:MazeF2',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MiyazakiA-v0',
    entry_point='my_mazes.envs:MiyazakiA',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MiyazakiB-v0',
    entry_point='my_mazes.envs:MiyazakiB',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='MazeF3-v0',
    entry_point='my_mazes.envs:MazeF3',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Sutton-v0',
    entry_point='my_mazes.envs:Sutton',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Woods1-v0',
    entry_point='my_mazes.envs:Woods1',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Woods14-v0',
    entry_point='my_mazes.envs:Woods14',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Woods100-v0',
    entry_point='my_mazes.envs:Woods100',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Woods101-v0',
    entry_point='my_mazes.envs:Woods101',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Woods101demi-v0',
    entry_point='my_mazes.envs:Woods101demi',
    max_episode_steps=100,
    nondeterministic=False
)

register(
    id='Woods102-v0',
    entry_point='my_mazes.envs:Woods102',
    max_episode_steps=100,
    nondeterministic=False
)