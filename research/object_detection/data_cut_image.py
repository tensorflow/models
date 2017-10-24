# -*- coding: utf-8 -*-
from PIL import Image
import sys
import os.path
from  datetime import *
import random
import time


# 图片压缩批处理
def cutImage(srcPath, dstPath, logo_width=None, logo_height=None, x_padding=0, y_padding=0):
    for filename in os.listdir(srcPath):
        # 如果不存在目的目录则创建一个，保持层级结构
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)

            # 拼接完整的文件或文件夹路径
        srcFile = os.path.join(srcPath, filename)
        dstFile = os.path.join(dstPath, filename)
        # print(srcFile)
        # print(dstFile)

        # 如果是文件就处理
        if os.path.isfile(srcFile):
            with Image.open(srcFile) as im:
                pix = im.load()
                img_width = im.size[0]
                img_height = im.size[1]
                # box = (IMAGE_X1, IMAGE_Y1, IMAGE_X2, IMAGE_Y2)  # 设定裁剪区域
                box = (
                    img_width - logo_width - x_padding,
                    y_padding,
                    img_width - x_padding,
                    y_padding + logo_height)  # 设定裁剪区域
                im.crop(box).save(dstFile)  # 保存图片


# 图片压缩批处理
def compressImage(srcPath, dstPath):
    for filename in os.listdir(srcPath):
        # 如果不存在目的目录则创建一个，保持层级结构
        if not os.path.exists(dstPath):
            os.makedirs(dstPath)

            # 拼接完整的文件或文件夹路径
        srcFile = os.path.join(srcPath, filename)
        dstFile = os.path.join(dstPath, filename)
        print(srcFile)
        print(dstFile)

        # 如果是文件就处理
        if os.path.isfile(srcFile):
            # 打开原图片缩小后保存，可以用if srcFile.endswith(".jpg")或者split，splitext等函数等针对特定文件压缩
            sImg = Image.open(srcFile)
            w, h = sImg.size
            dImg = sImg.resize(size=(w * 2, h * 2))  # 设置压缩尺寸和选项，注意尺寸要用括号
            dImg.save(dstFile)  # 也可以用srcFile原路径保存,或者更改后缀保存，save这个函数后面可以加压缩编码选项JPEG之类的
            print(dstFile + " compressed succeeded")

        # 如果是文件夹就递归
        if os.path.isdir(srcFile):
            compressImage(srcFile, dstFile)


if __name__ == '__main__':
    # gain_train_data()
    srcPath = r"E:\data_mining\data\east_ic_logo\train\BigLogo"
    dstPath = r"E:\data_mining\data\east_ic_logo\train\BigLogo_cut"

    class_name = os.path.basename(srcPath)
    if class_name == "BigLogo":
        logo_width = 160
        logo_height = 76
        x_padding = 20
        y_padding = 16
        print("big logo = " + class_name)
    else:
        print("small logo = " + class_name)
        logo_width = 96
        logo_height = 44
        x_padding = 10
        y_padding = 10

    cutImage(srcPath, dstPath, logo_width, logo_height, x_padding, y_padding)
