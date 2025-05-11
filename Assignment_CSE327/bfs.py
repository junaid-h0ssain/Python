from collections import deque

def bfs(graph, start_node):
    queue = deque([start_node])
    visited = set()
    traversal_order = []

    while queue:
        current_node = queue.popleft()
        if current_node not in visited:
            visited.add(current_node)
            # Add it to the traversal order
            traversal_order.append(current_node)
            # Add all unvisited neighbors to the queue
            for neighbor in graph[current_node]:
                if neighbor not in visited:
                    queue.append(neighbor)

    return traversal_order

"""The given graph is a representation of a directed graph using an adjacency list. Each key in the dictionary is a node, and its associated value is a list of nodes it is directly connected to by outgoing edges.

**Key-Value Pair:**

Keys ('5', '3', etc.) represent the nodes (or vertices) in the graph.

Values (e.g., ['3', '7']) are lists of nodes that the corresponding node has edges pointing to.
"""

# Example usage:
graph = {
   '5' : ['3','7'],
  '3' : ['2', '4'],
  '7' : ['8'],
  '2' : [],
  '4' : ['8'],
  '8' : []
}

start_node = '5'
print("BFS Traversal:", bfs(graph, start_node))

"""**Modify the BFS algorithm to skip nodes marked as obstacles. Given a graph and a list of obstacles, return the traversal order starting from a given node.**

graph = {
    'A': ['B', 'C'],

    'B': ['A', 'D', 'E'],

    'C': ['A', 'F'],

    'D': ['B'],

    'E': ['B', 'F'],

    'F': ['C', 'E']
}

obstacles = ['D', 'F']

start_node = 'A'

Traversal Order Skipping Obstacles: ['A', 'B', 'C', 'E']
"""

from collections import deque

def bfs_with_obstacles(graph, start_node, obstacles):
    # Initialize a queue for BFS
    queue = deque([start_node])
    # Set to keep track of visited nodes
    visited = set()
    # List to store the order of traversal
    traversal_order = []

    while queue:
        # Pop the first node from the queue
        current_node = queue.popleft()

        # If the node is an obstacle or already visited, skip it
        if current_node not in visited and current_node not in obstacles:
        # Mark the node as visited
            visited.add(current_node)
        # Add it to the traversal order
            traversal_order.append(current_node)

        # Add all unvisited, non-obstacle neighbors to the queue
            for neighbor in graph[current_node]:
                if neighbor not in visited and neighbor not in obstacles:
                    queue.append(neighbor)

    return traversal_order

# Example usage:
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

obstacles = ['D', 'F']
start_node = 'A'

print("Traversal Order Skipping Obstacles:", bfs_with_obstacles(graph, start_node, obstacles))