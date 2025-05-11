def dfs(graph, start):
    stack = [start]
    visited = set()
    traversal_order = []

    while stack:
        current = stack.pop()

        if current not in visited:
            visited.add(current)
            traversal_order.append(current)

            for i in reversed(graph[current]):
                if i not in visited:
                    stack.append(i)

    if len(visited) != len(graph):
        print("The graph is not connected.")
        print(f"Total nodes visited {len(visited)}, total nodes in graph {len(graph)}")
    else:
        print("The graph is connected.")
        print(f"Total nodes visited {len(visited)}, total nodes in graph {len(graph)}")
    
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
    '5':[]
}

start = '0'
print("DFS Traversal:", dfs(graph, start))