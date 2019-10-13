import requests, json, cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

#URL = 'http://127.0.0.1:5000/v1/models/addtest:predict'
URL = 'http://143.248.96.81:35005/api/predict'


img_path = './data/ki_corridor/2.png'
im1 = cv2.imread(img_path)

if im1.shape != [480,640]:
    im1 = cv2.resize(im1, (640, 480))

im1=cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
print(im1.shape)
rNum = 480
cNum = 640

strImgTest = '['
for r in range(rNum):
    strImgTest+='['
    for c in range(cNum):
        strTmp = '['+str((im1[r][c][0]))+','+str((im1[r][c][1]))+','+str((im1[r][c][2]))+']'
        strImgTest+=strTmp

        if c !=cNum-1 :
            strImgTest+=','
    strImgTest+=']'
    if r != rNum-1:
        strImgTest += ','
strImgTest+=']'

#plt.figure(figsize=(20, 15))
#plt.imshow(img)
#plt.show()

res = requests.post(url=URL, data={'image':strImgTest})
print(res.text)