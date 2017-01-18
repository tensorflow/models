from collections import Counter

with open("datadecoded") as datainput, open("vocab", 'w') as v:
    wordcount = Counter(datainput.read().split())
    for item in wordcount.items(): 
        print >>v, ("{} {}".format(*item))


