def minmax(tree,d,ismax):
    if d==0 or 'v'in tree:
        return tree.get('v','there is no tree')
    
    branch = tree.get('trunk',tree.get('branch1',tree.get('branch2',tree.get('branch3',[]))))

    if ismax:
        best=-float('inf')
        for leave in branch:
            val=minmax(leave,d-1,False)
            best=max(best,val)
        return best
    else:
        best=float('inf')
        for leave in branch:
            val=minmax(leave,d-1,True)
            best=min(best,val)
        return best

tree = {
    'trunk':[
        {
            'branch1':[
                {
                    'branch2':[
                        {
                            'branch3':[
                                {'v':4},
                                {'v':3},
                                {'v':5}
                            ],
                        },
                        {
                            'branch3':[
                                {'v':2},
                                {'v':1}
                            ]
                        }
                    ]
                },
                {
                    'branch2':[
                        {
                            'branch3':[
                                {'v':4},
                                {'v':2},
                                {'v':3}
                            ]
                        }
                    ]
                },
                {
                    'branch2':[
                        {
                            'branch3':[
                                {'v':6},
                                {'v':4}
                            ]
                        },
                        {
                            'branch3':[
                                {'v':7}
                            ]
                        },
                        {
                            'branch3':[
                                {'v':5},
                                {'v':2}
                            ]
                        }
                    ]
                }
            ]
        },
        {
            'branch1':[
                {
                    'branch2':[
                        {
                            'branch3':[
                                {'v':1},
                                {'v':9},
                                {'v':0}
                            ],
                        }
                    ]
                },
                {
                    'branch2':[
                        {
                            'branch3':[
                                {'v':4},
                                {'v':3}
                            ]
                        },
                        {
                            'branch3':[
                                {'v':0}
                            ]
                        }
                    ]
                },
                {
                    'branch2':[
                        {
                            'branch3':[
                                {'v':2},
                                {'v':8},
                                {'v':4}
                            ]
                        },
                        {
                            'branch3':[
                                {'v':3},
                                {'v':7}
                            ]
                        },
                        {
                            'branch3':[
                                {'v':5},
                                {'v':4},
                                {'v':1}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
}

print(minmax(tree,d=4,ismax=True))

