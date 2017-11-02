# -*- coding: utf-8 -*-
from PIL import Image
import sys
import os.path
from  datetime import *
import random
import time
import shutil

if __name__ == '__main__':
    # gain_train_data()
    srcPath = r"C:\Users\sunhongzhi\Desktop\delete"
    dstPath = r"E:\data_mining\data\east_ic_logo\train\SmallLogo"
    deleted_path = r"C:\Users\sunhongzhi\Desktop\deleted_small"

    for filename in os.listdir(srcPath):
        dstFile = os.path.join(dstPath, filename)
        deleted_file = os.path.join(deleted_path, filename)
        shutil.copyfile(dstFile, deleted_file)
        os.remove(dstFile)
        print("删除文件" + dstFile)
