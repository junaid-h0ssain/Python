input_list = ["apple", "bat", "car", "banana", "cat", "ball", "act"]

output = {}

for i in input_list:
    size = len(i)
    if size not in output:
        output[size] = []
    output[size].append(i)

print(f"{output}")
        