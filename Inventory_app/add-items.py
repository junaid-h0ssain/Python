import json
import os

db='db.json'

def add(name:str, price:float, amount:int, image:str):
    items = []
    item = {
        'itemname': name,
        'itemprice': price,
        'itemcount': amount,
        'itemimg': image
    }

    if os.path.exists(db):
        with open(db,'r') as file:
            items=json.load(file)

    items.append(item)

    with open(db,'w') as file:
        json.dump(items,file)

    print(f"Item '{item}' saved to {db}.")
    

# def view(item):
#     for i in range(len(item['itemname'])):
#         print(f"Name: {item['itemname'][i]}")
#         print(f"Price: {item['itemprice'][i]}")
#         print(f"Count: {item['itemcount'][i]}")
#         print(f"Image: {item['itemimg'][i]}")
#         print("-" * 20)

add('food', 23.45, 65, 'burger.jpg')
add('drink', 2.00, 120, 'soda.png')
#view(item)