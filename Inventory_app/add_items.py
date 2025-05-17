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

    duplicate = False
    for existing_item in items:
        if existing_item.get('itemname') == name:
            duplicate = True
            print(f"Item '{name}' already exists in {db}. Not adding duplicate.")
            break

    if not duplicate:
        items.append(item)
        with open(db,'w') as file:
            json.dump(items,file,indent=4)
        print(f"Item {item} saved to db successfully.")

def view():
    with open(db,'r') as file:
        items=json.load(file)
    
    list = (f'Item name\tItem price\tItem count\tItem image\n')
    list +='------------------------------------------------------------------------------------------------\n'
    for item in items:
        list += (f"{item['itemname']}\t\t"+
            f"{item['itemprice']}\t\t"+
            f"{item['itemcount']}\t\t"+
            f"{item['itemimg']}\n")
        list +='---------------------------------------------------------------------------------------------\n'

    return list
        

add('food', 23.45, 65, 'burger.jpg')
add('drink', 20.00, 120, 'soda.png')
add('soap',30.12,45,'soap.jpg')
add('paint',500.54,12,'paint.png')

