from collections import deque

def bfs_with_obstacles(graph, start, obstacle):
    queue = deque([start])
    visited = set()
    traversal_order = []

    while queue:
        current = queue.popleft()

        if current not in visited and current not in obstacle:
            visited.add(current)
            traversal_order.append(current)

            for i in graph[current]:
                if i not in visited and i not in obstacle:
                    queue.append(i)

    return traversal_order

# Example usage:
graph = {
    '0': ['1', '9'],
    '1': ['6', '3'],
    '2': [],
    '3': ['4','2'],
    '4': [],
    '6': ['8'],
    '9': ['6','13'],
    '8': [],
    '13': [],
}

obstacle = ['6']
start = '0'

print("Traversal Order Skipping Obstacles:", bfs_with_obstacles(graph, start, obstacle))