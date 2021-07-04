from .push import PushEnv
from .ball import BallEnv
from .maze_utils import get_maze_structure

def create_maze(domain, task, maze_structure=None):

    if maze_structure is None:
        maze_structure = get_maze_structure(task=task)


    if domain == "ball":
        return BallEnv(maze_structure=maze_structure)
    elif domain == "push":
        return PushEnv(maze_structure=maze_structure)
    else:
        raise ValueError("Unknown fetch domain.")