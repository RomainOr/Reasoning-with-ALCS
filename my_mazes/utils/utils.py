import networkx as nx

from ..maze import Maze

def create_graph(env):
    maze = env.maze

    # Create uni-directed graph
    g = nx.Graph()

    # Add nodes
    for x in range(0, maze.max_x):
        for y in range(0, maze.max_y):
            if maze.is_path(x, y):
                g.add_node((x, y), type='path')
            if maze.is_reward(x, y):
                g.add_node((x, y), type='reward')
            if maze.is_obstacle(x, y):
                g.add_node((x, y), type='obstacle')

    # Add edges
    path_nodes = [cords for cords, attribs
        in g.nodes(data=True) if attribs['type'] == 'path' or attribs['type'] == 'obstacle']

    for n in path_nodes:
        neighbour_cells = Maze.get_possible_neighbour_cords(*n)
        allowed_cells = [c for c in neighbour_cells
            if maze.is_path(*c) or maze.is_reward(*c)]
        edges = [(n, dest) for dest in allowed_cells]
        g.add_edges_from(edges)

    return g
