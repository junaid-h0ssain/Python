my_list = ['hello', 'world', 'python', 'is', 'awesome']
letter_counts = {}  # Initialize an empty dictionary to store the counts

for word in my_list:
    for letter in word:
        if letter in letter_counts:
            letter_counts[letter] += 1
        else:
            letter_counts[letter] = 1

print(letter_counts)

