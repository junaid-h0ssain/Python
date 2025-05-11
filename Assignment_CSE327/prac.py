def dfs(graph, start_node):
    stack = [start_node]
    visited = set()
    traversal_order = []

    while stack:
        current_node = stack.pop()

        if current_node not in visited:
            visited.add(current_node)
            traversal_order.append(current_node)

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
    '8': ['5'],
    '6':[]
}

start_node = '5'
print("DFS Traversal:", dfs(graph, start_node))