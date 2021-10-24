import json
import scipy.io as sio
import os

file = os.path.join("D:\\", "Programming", "pix3d", "model", "bookcase", "IKEA_BILLY_1", "model.obj")

vertices = []
faces = []

obj_file = open(file, 'r')
lines = obj_file.readlines()

for line in lines:
    lineID = line[0:2]
    
    if lineID == "v ":
        vertex = lines[1:].split(" ")
        
        for i, v in enumerate(vertex):
            vertex[i] = float(v)

        vertices.append(vertex)

    if lineID == "f ":

        face = lines[1:].split(" ")
        
        for i, f in enumerate(face):
            face[i] = [int(x) for x in f.split("/")]

        faces.append(face)


print(vertices, faces)


"""
dir = os.path.join("D:\\", "Programming", "pix3d", "model", "bookcase", "IKEA_BILLY_1", "voxel.mat")

test = sio.loadmat(dir)

print(test["voxel"])
"""



"""

load = json.load(open("D:\Programming\pix3d\pix3d.json"))

print(load[0])

image = load[0]

l = {"img": image["img"], "category": image["category"], "img_size": image["img_size"], "2d_keypoints": image["2d_keypoints"],
     "mask": image["mask"], "img_source": image["img_source"], "model": image["model"], "model_raw": image["model_raw"],
     "model_source": image["model_source"], "3d_keypoints": image["3d_keypoints"], "voxel": image["voxel"], "rot_mat": image["rot_mat"],
     "trans_mat": image["trans_mat"], "focal_length": image["focal_length"], "cam_position": image["cam_position"],
     "inplane_rotation": image["inplane_rotation"], "truncated": image["truncated"], "occluded": image["occluded"],
     "slightly_ocluded": image["slightly_occluded"], "bbox": image["bbox"]}

print(l)
"""

"""
for photo in load:
    print(photo)
    print("\n")
"""
