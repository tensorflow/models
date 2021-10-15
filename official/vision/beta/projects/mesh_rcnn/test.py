import json

load = json.load(open("D:\Programming\pix3d\pix3d.json"))

print(load[0])

for photo in load:
    print(photo)
    print("\n")