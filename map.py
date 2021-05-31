import numpy as np

def generate_random_map(size=8, p=0.8, min_dist=3):
    """Generates a random valid map (one that has a path from start to goal)

    + is the endgoal
    r is the start position
    0 is free space
    1 is a wall

    :param size: size of each side of the grid
    :param p: probability that a tile is frozen
    """
    valid = False

    # BFS to check that it's a valid path.
    def is_valid(res, start):
        queue, visited = [], set()
        queue.append((start, 1))
        visited.add(start)
        while queue:
            (r, c), dist = queue.pop(0)
            directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            for x, y in directions:
                r_new = r + x
                c_new = c + y
                if r_new < 0 or r_new >= size or c_new < 0 or c_new >= size:
                    continue
                if res[r_new][c_new] == 'G':
                    if dist + 1 >= min_dist: 
                        print("distance", dist+1)
                        return [True, dist+1]
                    else:
                        return False
                if res[r_new][c_new] not in '#H':
                    if (r_new, c_new) not in visited:
                        visited.add((r_new, c_new))
                        queue.append(((r_new, c_new), dist+1))
        return False

    while not valid:
        p = min(1, p)
        res = np.random.choice(['F', 'H'], (size, size), p=[p, 1-p])
        F_indices = [(i,j) for i in range(size) for j in range(size) if res[i][j]=='F']
        start_idx = F_indices[np.random.randint(len(F_indices))]
        res[start_idx[0]][start_idx[1]] = 'S'
        F_indices = [(i,j) for i in range(size) for j in range(size) if res[i][j]=='F']
        goal_idx = F_indices[np.random.randint(len(F_indices))]
        res[goal_idx[0]][goal_idx[1]] = 'G'
        valid = is_valid(res, start_idx)
    replace_dict = {
        'S' : 'r',
        'G' : '+',
        'F' : 0,
        'H' : 1,
    }
    # Put a wall around the maze and replace the symbols.
    return [[1] * (size+2)] + [[1] + [replace_dict[item] for item in row] + [1] for row in res] + [[1] * (size+2)], valid[1]

if __name__ == "__main__":
    if False:
        maze = generate_random_map(size=4)
        # print(maze)
        print('\n')
        for row in maze:
            print(row)
    else:
        import pickle
        mazes = []
        distances = []
        for i in range(501):
            maze, dist = generate_random_map(size=4)
            mazes.append(maze)
            distances.append(dist)
        print(np.mean(distances))
        pickle.dump({'mazes': mazes, 'distances':distances}, open('RandomMazes.pkl', 'wb'))
    # print("\n", maze[0], "\n", maze[1], "\n", maze[2], "\n", maze[3])