def dfs(graph, start_node):
    # Initialize a stack for DFS
    stack = [start_node]
    # Set to keep track of visited nodes
    visited = set()
    # List to store the order of traversal
    traversal_order = []

    while stack:
        # Pop the last node from the stack
        current_node = stack.pop()
        # If the node has not been visited
        if current_node not in visited:
            # Mark it as visited
            visited.add(current_node)
            # Add it to the traversal order
            traversal_order.append(current_node)

            # Add all unvisited neighbors to the stack
            # Reverse the neighbors to maintain the correct order
            for neighbor in reversed(graph[current_node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    return traversal_order

# Example usage:
graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}
start_node = '5'
print("DFS Traversal:", dfs(graph, start_node))

"""**Problem 1: Check if a graph is connected**

**Problem 2: Count Connected Components**
"""

def dfs(graph, start_node):
    # Initialize a stack for DFS
    stack = [start_node]
    # Set to keep track of visited nodes
    visited = set()
    # List to store the order of traversal
    traversal_order = []

    while stack:
        # Pop the last node from the stack
        current_node = stack.pop()

        # If the node has not been visited
        if current_node not in visited:
            # Mark it as visited
            visited.add(current_node)
            # Add it to the traversal order
            traversal_order.append(current_node)

            # Add all unvisited neighbors to the stack
            # Reverse the neighbors to maintain the correct order
            for neighbor in reversed(graph[current_node]):
                if neighbor not in visited:
                    stack.append(neighbor)

    if len(visited) != len(graph):
        print("The graph is not connected.")
        print(len(visited))
        return traversal_order
    else:
        print("The graph is connected.")
        print(len(visited))
        return traversal_order

# Example usage:
graph = {
    '5': ['3', '7'],
    '3': ['2', '4'],
    '7': ['8'],
    '2': [],
    '4': ['8'],
    '8': []
}

start_node = '5'
print("DFS Traversal:", dfs(graph, start_node))