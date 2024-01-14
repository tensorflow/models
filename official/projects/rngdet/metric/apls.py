import numpy as np
import os
import argparse
import json

apls = []
name_list = os.listdir(f'../segmentation/tests/results/apls')
name_list.sort()
for file_name in name_list :
    try:
        with open(f'../segmentation/tests/results/apls{file_name}') as f:
            lines = f.readlines()
        print(file_name,lines[0].split(' ')[-1][:-2])
        apls.append(float(lines[0].split(' ')[-1][:-2]))
    except:
        break
print('APLS',np.mean(apls))
with open(f'../segmentation/tests/score/apls.json','w') as jf:
    json.dump({'apls':apls,'final_APLS':np.mean(apls)},jf)