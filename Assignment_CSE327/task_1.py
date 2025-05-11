list = [12,4,45,6,13,8,0,20]
output = []

for i in list:
    if i>10:
        output.append(i)

output.sort()
print('Numbers greater than 10 : ')
print(output)