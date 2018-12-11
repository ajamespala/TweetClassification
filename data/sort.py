import random
with open('test4.txt','r') as source:
    data = [ (random.random(), line) for line in source ]
data.sort()
with open('test4.txt','w') as target:
    for _, line in data:
        target.write( line )
