import random
with open('data15.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('data15.txt','w') as target:
    for _, line in data:
        target.write( line )
