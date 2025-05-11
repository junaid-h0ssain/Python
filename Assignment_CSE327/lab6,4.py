def minimax(node, depth, is_max):
    if depth==0 or 'value' in node:
        return node.get('value','There is no tree')

    children = node.get('children', node.get('subchild',[]))
    
    if is_max:
        best = -float('inf')
        for child in children:
            val = minimax(child,depth-1,False)
            best = max((best,val))
        return best
    else:
        best = float('inf')
        for child in children:
            val = minimax(child,depth-1,True)
            best = min(best,val)
        return best

tree = {
    'children' :[
        {
            'subchild':[
                {
                    'subchild':[
                        {'value':3},
                        {'value':2}
                    ]
                },
                {
                    'subchild':[
                        {'value':5},
                        {'value':6}
                    ]
                }
            ]
        },
        {
            'subchild':[
                {
                    'subchild':[
                        {'value':1},
                        {'value':2}
                    ]
                },
                {
                    'subchild':[
                        {'value':4},
                        {'value':3}
                    ]
                }
            ]
        }
    ]
}

# tree = {
#     'value':420
# }

print(minimax(tree,depth=0,is_max=True))