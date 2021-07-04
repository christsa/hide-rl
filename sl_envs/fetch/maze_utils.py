import map as map_generator

X = 0
O = 'O'
T = '+'
EMPTY_MAZE_STRUCTURE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, X, X, X, X, X, X, X, 1],
    [1, X, 0, 0, 0, 0, 0, X, 1],
    [1, X, 0, 0, 0, 0, 0, X, 1],
    [1, X, 0, 0, 0, 0, 0, X, 1],
    [1, X, 0, 0, 0, 0, 0, X, 1],
    [1, X, 0, 0, 0, 0, 0, X, 1],
    [1, X, X, X, X, X, X, X, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]

EMPTY_MAZE_STRUCTURE_NO_WALLS = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, X, X, X, X, X, X, X, 0],
    [0, X, 0, 0, 0, 0, 0, X, 0],
    [0, X, 0, 0, 0, 0, T, X, 0],
    [0, X, O, 0, 0, 0, 0, X, 0],
    [0, X, 0, 0, 0, 0, 0, X, 0],
    [0, X, 0, 0, 0, 0, 0, X, 0],
    [0, X, X, X, X, X, X, X, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]


ORIG_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, X, X, 1, 1, X, X, X, 1],
    [1, X, O, 1, 1, T, 0, X, 1],
    [1, X, 0, 1, 1, 1, 0, X, 1],
    [1, X, 0, 1, 1, 1, 0, X, 1],
    [1, X, 0, 0, 0, 1, 0, X, 1],
    [1, X, 0, 0, 0, 0, 0, X, 1],
    [1, X, X, X, X, X, X, X, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]

SIMPLE_MAZE = [
    [0, 0, 0, 1, 1, 1, 1, 1, 1],
    [0, X, X, 1, 1, X, X, X, 1],
    [0, 0, O, 1, 1, T, 0, X, 1],
    [0, X, 0, 1, 1, 1, 0, X, 1],
    [0, X, 0, 1, 1, 1, 0, X, 0],
    [0, X, 0, 0, 0, 1, 0, X, 0],
    [0, X, 0, 0, 0, 0, 0, X, 0],
    [0, X, X, X, X, X, X, X, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

SIMPLE_RANDOM_MAZE = [
    [0, X, X, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, X, 0, 0, 0, 0, 0, 0, 1],
    [0, T, 0, 0, 0, 0, 0, 0, 1],
    [0, 1, 0, 0, 0, 1, 0, X, 0],
    [0, 1, 1, 0, 0, 1, 1, O, 0],
    [1, 1, 0, X, X, 1, 1, X, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0],
]

# SIMPLE_RANDOM_MAZE = [
#     [0, 0, 0, 0, 0, 0, 1, 0, 0],
#     [0, X, X, 1, T, 0, 0, 0, 1],
#     [0, 0, 0, 0, 0, 0, 0, 0, 1],
#     [0, X, 1, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 0, 0, 0],
#     [0, 0, 1, 1, 1, 1, 0, X, 0],
#     [0, 0, 0, O, 0, 0, 0, 0, 0],
#     [0, 0, 0, X, 1, 0, 0, X, 0],
#     [0, 0, 0, 0, 1, 0, 0, 0, 0],
# ]

ORIG_RANDOM_MAZE = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, X, X, 1, 1, X, X, X, 1],
    [1, X, O, 1, 1, 0, 0, X, 1],
    [1, X, 0, 1, 1, 1, 0, X, 1],
    [1, X, 0, 1, 1, 1, 0, X, 1],
    [1, X, 0, 0, 0, 1, 0, X, 1],
    [1, X, 0, 0, 0, 0, 0, X, 1],
    [1, X, X, X, X, X, X, X, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAZES = {
    'simple' : SIMPLE_RANDOM_MAZE,
    'orig' : ORIG_MAZE, 
    'origrandom' : ORIG_RANDOM_MAZE, 
    'simplerandom' : SIMPLE_RANDOM_MAZE,
    'empty' : EMPTY_MAZE_STRUCTURE, 
    'nowalls' : EMPTY_MAZE_STRUCTURE_NO_WALLS,
    'random_no_fence': map_generator.generate_random_map(size=7, fence_symbol=0)[0],
    'random_with_fence': map_generator.generate_random_map(size=7, fence_symbol=1)[0]
}

def get_maze_structure(task):
    return MAZES.get(task)
