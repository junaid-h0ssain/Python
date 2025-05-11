h = {
    'A':12,
    'B':19,
    'C':13,
    'D':6,
    'E':6,
    'F':4,
    'G':0,
    'H':3,
}

graph = {
    'A': {'B': 6, 'D': 6},
    'B': {'A': 6, 'C': 5, 'E': 10},
    'C': {'B': 5, 'F': 7},
    'D': {'A': 6, 'H': 4},
    'E': {'B': 10, 'F': 6},
    'F': {'C': 7, 'E': 6, 'H': 6},
    'G': {'H': 3},
    'H': {'D': 4, 'F': 6, 'G': 3},
}

def a_star_search(start, goal):
    open_set = {start: h[start]}
    came_from = {}
    path_cost = {node: float('inf') for node in graph.keys()}
    path_cost[start] = 0

    while open_set:
        current = min(open_set, key=lambda x: open_set[x])

        if current == goal:
            print("Path cost:", path_cost[current])
            return reconstruct_path(came_from, current)

        del open_set[current]

        for neighbor, dist in graph[current].items():
            tentative_path_cost = path_cost[current] + dist
            if tentative_path_cost < path_cost[neighbor]:
                came_from[neighbor] = current
                path_cost[neighbor] = tentative_path_cost
                f_score = tentative_path_cost + h[neighbor]
                open_set[neighbor] = f_score

    return None

def reconstruct_path(came_from, goal):
    path = [goal]
    while goal in came_from:
        goal = came_from[goal]
        path.insert(0, goal)
    return path

path = a_star_search('A', 'G')
print("Path found by A*:", path)