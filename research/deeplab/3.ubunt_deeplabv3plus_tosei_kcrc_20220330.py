# -*- coding: utf-8 -*-

# DeepLabV3+ to DeepSurv in train tosei test kcrc Docker
# 肺野を抽出するDL→出た物をDL　ラベルは①肺野ありバージョン、②無しバージョンをやる。
# (DL that extracts the lung fields → DL of what comes out.The labels are (1) version with lung fields and (2) version without lung fields.)

# # tensorflow-gpuのDockerをインストール
# (Install Docker for tensorflow-gpu)
# docker pull tensorflow/tensorflow:1.15.5-gpu-py3


# 1.1 DICOM画像をPNGへ事前処理（陶生、神奈川、Normal）
# 1.2 肺、肺外を検出する2クラス問題（LungOrNot）でDLモデリング
# Translated
# 1.1 Preprocessing DICOM images to PNG (Tosei, Kanagawa, Normal)
# 1.2 DL modeling with two-class problem (LungOrNot) to detect lung and extra-lung

# 2.1 学習と結果 DL
    # 2.1.1 陶生のエクセルデータから、IPF or Notで層別化したTrain, Val lst作成, KCRCデータも取得
        # TF Recordを生成
            # 2.1.2 datagenerator.pyの書き換え
            # 2.2.3 実行
                # image folder, semantic_segmentation_folder, list_folder, output_dir を指定
    # 2.2.1 Training & Val DL
        # train_logdir, dataset_dirを指定
    # 2.3.1 eval log
    # 2.4.1 vislog
    # 2.5.1 test log
# Translated
# 2.1 Learning and results DL
    # 2.1.1 Created Train, Val lst stratified by IPF or Not, and obtained KCRC data from Tosei's Excel data
        # Generate TF Record
            # 2.1.2 Rewriting datagenerator.py
            # 2.2.3 execution
                # image folder, semantic_segmentation_folder, list_folder, output_dir specify
    # 2.2.1 Training & Val DL
        # train_logdir, dataset_dir specify
    # 2.3.1 eval log
    # 2.4.1 vislog
    # 2.5.1 test log




import time
import os  # needed navigate the system to get the input data
import sys
import glob
import re
import cv2
import shutil
import random
import math
import pandas as pd
import numpy as np
import six, pickle
import itertools
import SimpleITK as sitk
# import matplotlib.pyplot as plt
from openpyxl import load_workbook
from PIL import Image, ImageOps
import pandas as pd
from pathlib import Path
# import tensorflow as tf

# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split


# # コマンドライン引数を受け取る
# # receive command line arguments
# args = sys.argv


def _min_max_(X,axis=None):
    # 正規化関数
    # normalization function -> min-max scaling
    min = X.min(axis=axis, keepdims=True)
    max = X.max(axis=axis, keepdims=True)
    result = (X-min)/(max-min)
    return result



def _conv_dcm_(dcmfullpath):

    if not '.tif' in dcmfullpath:
        # DICOM画像をPNG画像にする
        # pydicomはDICOMの書式のゆらぎがあるとReadできない >> sitk
        # Convert DICOM image to PNG image
        # pydicom cannot be read if there is a fluctuation in the DICOM format >> sitk
        xITK = sitk.ReadImage(dcmfullpath)
        # ITK形式をndarray形式に変換 dtype=uint8
        # Convert ITK format to ndarray format dtype=uint8
        x0 = sitk.GetArrayFromImage(xITK)
        # [1,512,512] >> [512,512]へ >> cv2にかけるためにfloat３２に変換
        # [1,512,512] >> to [512,512] >> convert to float32 to multiply by cv2
        x = np.float32(np.squeeze(x0))
    else:
        # Tif画像をPNG画像にする
        # 16bit画像をopen CVで読み込む
        # Convert Tif image to PNG image
        # Load 16bit image with open CV
        x = cv2.imread(dcmfullpath, cv2.IMREAD_ANYDEPTH)


    # 骨より大きいCT値は骨の値へ, low以下はlowへ
    # CT value larger than bone goes to bone value, below low goes to low
    bone = 350
    low = -round(1024*1.2-bone)
    # -1250 >> -879 @2022/3/27

    # numpy.int >> python intへ
    # numpy.int >> to python int
    a = x[0,0].item()
    # 空気はHU -1000であることを利用
    # Use the fact that air is HU -1000
    air = x[5:20,256-5:256+5].mean()
    if (x.min()>-500):
    # if (type(a)==int and x.min()>-500):
        bone = bone - (-1000 - air)
        low = low - (-1000 - air)

    # if not x is np.float32:
        # x = np.float32(x)

    x2 = np.clip(x, low, bone)

    # 正規化して、uint8 PNGにするために255をかける
    # Normalize and multiply by 255 to get a uint8 PNG
    x3 = _min_max_(x2) * 255
    # python pngはuint8しか受け付けないので、変換して保存
    # python png only accepts uint8, so convert and save
    x4 = x3.astype(np.uint8)
    return(x4)


def _conv_mskgray_(mskfullpath):
    # Mask画像をGrayScale画像に変換する 白（肺）と黒（背景）を入れ替える
    # Convert Mask image to GrayScale image Swap white (lungs) and black (background)

    # Target Index (Index to Grayだけど、Indexと同様に処理される)
    # Target Index (Index to Gray, but treated the same as Index)
    # ' Lung -> 1
    # ' IPF -> 4
    # ' NonIPF -> 9
    # ' background ->> 0

    # openCOVID19
    # 1 -> Lung
    # 2 -> background ->> 0


    with Image.open(mskfullpath) as x:
        if not 'covid19' in mskfullpath:
            # RGBカラー画像をndarrayにする （**not index color**)
            # Convert RGB color image to ndarray (**not index color**)
            gray = x.convert("L")

            # [0;29;76;255] -> 肺外; nonIPF; IPF; 肺 [0; 4; 9; 1] に変換。
            # [0;29;76;255] -> convert to extrapulmonary; nonIPF; IPF; pulmonary [0; 4; 9; 1].
            gray = np.array(gray)
            gray[gray==29] = 4
            gray[gray==76] = 9
            gray[gray==255] = 1
            # gray[gray==29] = 2
            # gray[gray==76] = 3
            # gray[gray==255] = 1

            # # gray[gray==0] = 255 -> 肺外削除のため最後に実行 (Last performed for extrapulmonary removal)
            # gray = Image.fromarray(gray)

            # # cv2にかけるためにfloat３２に変換
            # # convert to float32 for multiplication with cv2
            # gray = np.float32(gray)

        else:
            # インデックスカラー画像をndarrayにする 
            # convert index color image to ndarray
            # 1 -> background,  2 -> Lung
            gray = np.array(x)
            gray[gray==2] = 0


        return(gray)


def _fgauss_(img,msk):
    # gaussfilt rand制御(2)
    # gaussfilt rand control(2)
    gaussrand = 0.3 + random.random()
    kerseize = 2 * math.ceil(2*gaussrand) + 1 # matlabの結果に合わせる
    img2 = cv2.GaussianBlur(img, (kerseize,kerseize), gaussrand)
    # cv2.GaussianBlur(インプット画像, カーネルサイズ, ガウス関数のσ)
    return(img2, msk)

def _fmedian_(img, msk):
    img2 = cv2.medianBlur(img, 3)
    return(img2, msk)

def _ftrans_(img, msk):
    tratex2 = random.randint(-10, 10) # x方向の移動 (move in x direction)
    tratey2 = random.randint(-10, 10) # y方向の移動 (move in x direction)
    M = np.float32([[1,0,tratex2],[0,1,tratey2]]) # 移動量
    img2 = cv2.warpAffine(img,M,(512,512), borderValue=int(img[1,1]))
    msk2 = cv2.warpAffine(msk,M,(512,512), borderValue=int(msk[1,1]), flags=cv2.INTER_NEAREST)
    # msk画像で回転すると、値がflaotで補間されるので、インデックス処理がおかしくなる。
    # デフォルトは線形補間なので、
    # flags=cv2.INTER_NEARESTで補間を最近傍にすれば問題解決)
    # Rotating with an msk image results in incorrect indexing as the values are interpolated with floats.
    # default is linear interpolation, so
    # Problem solved by setting flags=cv2.INTER_NEAREST to nearest neighbor)
    return(img2, msk2)

def _frand_(img, msk):
    # rand回転
    rand_c = random.randint(0, 365)
    M = cv2.getRotationMatrix2D((256,256), rand_c, 1)
    img2 = cv2.warpAffine(img, M, (512, 512), borderValue=int(img[1,1]))
    msk2 = cv2.warpAffine(msk, M, (512, 512), borderValue=int(msk[1,1]), flags=cv2.INTER_NEAREST)
    return(img2, msk2)

def _frotate_(img, msk):
    # % %上下左右反転(5)
    # % %Flip up/down/left/right(5)
    rand_rot = random.randint(-1, 1)
    img2 = cv2.flip(img, rand_rot)
    msk2 = cv2.flip(msk, rand_rot)
    return(img2, msk2)


def _pngsave_(X, Y, outputpngfolder, outputmskfolder, file_name, repnum):
    # PNG save
    path_dcmpng_temp = outputpngfolder + file_name + '_e' + str(repnum) + ".png"
    cv2.imwrite(path_dcmpng_temp, X)

    # cv2.imwrite(outputmskfolder + file_name + '_e' + str(repnum) + ".png", Y)
    path_mskgray_temp = outputmskfolder + file_name + '_e' + str(repnum) + ".png"
    _save_annotation(Y, path_mskgray_temp)

def _pngsavetif_(X, Y, outputpngfolder, outputmskfolder, file_name, repnum):
    path_dcmpng_temp = outputpngfolder + file_name + '_e' + str(repnum) + ".png"
    cv2.imwrite(path_dcmpng_temp, X)
    cv2.imwrite(outputmskfolder + file_name + '_e' + str(repnum) + ".png", Y)


def _DCMtoPNG_(dcmfolder, mskfolder, outputpngfolder, outputmskfolder):
    
    # # *******************
    # dcmfolder = toseidcmpath
    # mskfolder = toseimskpath
    # outputpngfolder = toseiaugpngpath
    # outputmskfolder = toseiaugmskpath
    # # *******************

    pkl_file = os.path.join(outputmskfolder, "exist_mask.pkl")
    exist_file_names = list()

    dcmlst = os.listdir(dcmfolder)
    for file_num, file_name in enumerate(dcmlst):

        file_name = file_name.replace('.dcm','')
        exist_file_names.append("tosei" + file_name.split("_")[0])
        path_dcm = dcmfolder + file_name
        if 'KCRC' in file_name:
            path_dcm = path_dcm + '.dcm'

        path_dcmpng = outputpngfolder + file_name + ".png"
        path_msk = mskfolder + file_name + ".png"
        path_mskgray = outputmskfolder + file_name + ".png"


        # 2ファイルとも存在していたら変換する
        # Convert if both files exist
        if os.path.isfile(path_dcm) & os.path.isfile(path_msk):
            # dcm to png convert and save
            pngx = _conv_dcm_(path_dcm)
            mskx = _conv_mskgray_(path_msk)


            # mskx[mskx==0] = 255
            # mskx[mskx==1] = 0


            # # *****************************
            # # augmentationなしで確認する！ PNG確認するために。
            # cv2.imwrite(path_dcmpng, pngx)
            # cv2.imwrite(path_mskgray, mskx)
            # # *****************************

            _pngsave_(pngx, mskx, outputpngfolder, outputmskfolder, file_name, 0)

            # gausss or median -> rotate or flip -> trans
            for repnum in range(1,4):
                # 別々に受ける必要がある！
                # You have to take them separately!
                k = random.randint(0, 2)
                if  k == 0:
                    X, Y = (pngx,mskx)
                elif k == 1:
                    X, Y = _fgauss_(pngx,mskx)
                elif k == 2:
                    X, Y = _fmedian_(pngx,mskx)

                k2 = random.randint(0, 2)
                if  k2 == 0:
                    X1, Y1 = (X,Y)
                elif k2 == 1:
                    X1, Y1 = _frand_(X,Y)
                elif k2 == 2:
                    X1, Y1 = _frotate_(X,Y)

                X2, Y2 = _ftrans_(X1,Y1)
                # X2 = _conv_dcm2_(X2)


                _pngsave_(X2, Y2, outputpngfolder, outputmskfolder, file_name, repnum)
                # # PNG save
                # path_dcmpng_temp = outputpngfolder + file_name + '_e' + str(repnum) + ".png"
                # cv2.imwrite(path_dcmpng_temp, X2)

                # # cv2.imwrite(outputmskfolder + file_name + '_e' + str(repnum) + ".png", Y2)
                # path_mskgray_temp = outputmskfolder + file_name + '_e' + str(repnum) + ".png"
                # _save_annotation(Y2, path_mskgray_temp)

        else:
            print("None existing file {} - {} {} - {}".format(path_dcm, os.path.isfile(path_dcm), path_msk, os.path.isfile(path_msk)))

    # save existing file
    exist_file_names = list(set(exist_file_names))
    save_data = dict(exist_files=exist_file_names)
    with open(pkl_file, "wb") as handle:
        pickle.dump(save_data, handle, pickle.HIGHEST_PROTOCOL)


def _TiftoPNG_(dcmfolder, mskfolder, outputpngfolder, outputmskfolder):
    # k枚に一枚を使用する
    # use one in k
    k = 5
    #'open data sourceのTifからPNGへ変換、Maskも一緒。割合はK枚に１枚
    # Convert Tif from open data source to PNG, along with Mask. The ratio is 1 in K
    dcmlst = glob.glob(dcmfolder + '**/IM*.tif')
    
    # 症例ごとにファイルを抽出する
    # extract files for each case
    ptslst = [re.findall('patient[0-9]{3}',s) for s in dcmlst]
    ptslst = list(set(sum(ptslst,[])))
    for pts in ptslst:
        dcmlst_pts = [s for s in dcmlst if pts in s]
        for fullpath in dcmlst_pts[::5]:
            filename = "".join([s.replace('patient','') for s in re.findall('patient[0-9]{3}',fullpath)]) + '_' + os.path.splitext(os.path.basename(fullpath))[0].replace('IM00','')

            path_dcm = fullpath
            path_dcmpng = outputpngfolder + filename + '.png'
            path_msk = fullpath.replace(dcmfolder, mskfolder).replace('.tif','.png')
            path_mskgray = path_dcmpng.replace(outputpngfolder, outputmskfolder)


            # 2ファイルとも存在していたら変換する
            # Convert if both files exist
            if os.path.isfile(path_dcm) & os.path.isfile(path_msk):
                # dcm to png convert and save
                pngx = _conv_dcm_(path_dcm)
                mskx = _conv_mskgray_(path_msk)


                # mskx[mskx==0] = 255
                # mskx[mskx==1] = 0


                # # *****************************
                # # *****************************
                # # augmentationなしで確認する！ PNG確認するために。
                # # Check without augmentation! to check the PNG.
                # cv2.imwrite(path_dcmpng, pngx)
                # cv2.imwrite(path_mskgray, mskx)
                # # *****************************
                # # *****************************

                _pngsavetif_(pngx, mskx, outputpngfolder, outputmskfolder, filename, 0)

                # gusss or median -> rotate or flip -> trans
                for repnum in range(1,4):
                    # 別々に受ける必要がある！
                    # You have to take them separately!
                    k = random.randint(0, 2)
                    if  k == 0:
                        X, Y = (pngx,mskx)
                    elif k == 1:
                        X, Y = _fgauss_(pngx,mskx)
                    elif k == 2:
                        X, Y = _fmedian_(pngx,mskx)

                    k2 = random.randint(0, 2)
                    if  k2 == 0:
                        X1, Y1 = (X,Y)
                    elif k2 == 1:
                        X1, Y1 = _frand_(X,Y)
                    elif k2 == 2:
                        X1, Y1 = _frotate_(X,Y)

                    X2, Y2 = _ftrans_(X1,Y1)
                    # X2 = _conv_dcm2_(X2)

                    _pngsavetif_(X, Y, outputpngfolder, outputmskfolder, filename, repnum)
                    # # PNG save
                    # path_dcmpng_temp = outputpngfolder + filename + '_e' + str(repnum) + ".png"
                    # cv2.imwrite(path_dcmpng_temp, X2)
                    # cv2.imwrite(outputmskfolder + filename + '_e' + str(repnum) + ".png", Y2)

            else:
                print(fullpath + 'は存在しません！！！')


def _remove_colormap(filename):
    return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
  pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
  grayimg = pil_image.convert("L")
  grayimg.save(filename)
  # with tf.io.gfile.GFile(filename, mode='w') as f:
    # pil_image.save(f, 'PNG')


def _IndexGray_(inputmskfolder):
    outfolder = str(Path(inputmskfolder).parent) + '/SegClassRaw'
    os.makedirs(outfolder, exist_ok=True)
    annotations = glob.glob(os.path.join(inputmskfolder,'*.png'))

    for annotation in annotations:
        raw_annotation = _remove_colormap(annotation)
        filename = os.path.basename(annotation)
        _save_annotation(raw_annotation, str(outfolder) + '/' + filename)


def _loadxlsx_(filepath,sheetname):
    wb = load_workbook(filename = filepath)
    ws = wb[sheetname]
    data = ws.values
    data = list(data)
    return pd.DataFrame(data[1:], columns=data[0])


def _distlibution_openlst_(datasetpath, outlstfolder, trainsize, msksetpath):

    # データセットをランダムに分割したリストを作成する
    # Create a list of randomly split datasets
    dcmlist = os.listdir(datasetpath)
    ptlist = [re.sub('_[0-9]{3}_e[0-9]{1}.png', '', s) for s in dcmlist]
    ptlist = list(set(ptlist))
    num_bunkatu = int(len(ptlist) * trainsize)
    random.shuffle(ptlist)

    # train list
    f = open(outlstfolder+'train.txt', 'w', encoding='utf-8')
    for i in ptlist[:num_bunkatu]:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    # val list
    f = open(outlstfolder + 'val.txt', 'w', encoding='utf-8')
    for i in ptlist[num_bunkatu:]:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    # trainval list
    f = open(outlstfolder + 'trainval.txt', 'w', encoding='utf-8')
    for i in ptlist:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    print('**********************************')
    print('**********************************')
    print('dataset number of {}'.format(outlstfolder))
    df = pd.read_table(outlstfolder + 'train.txt', header=None)
    print('train n = : {}'.format(df.shape[0]))
    df = pd.read_table(outlstfolder + 'val.txt', header=None)
    print('val n = : {}'.format(df.shape[0]))
    df = pd.read_table(outlstfolder + 'trainval.txt', header=None)
    print('trainval n = : {}'.format(df.shape[0]))

    path1 = Path(outlstfolder)
    print('--image_folder= {}'.format(datasetpath))
    print('--semantic_segmentation_folder= {}'.format(str(Path(msksetpath).parent) + '/SegClassRaw'))
    print('--list_folder= {}'.format(outlstfolder))
    print('--output_dir= {}'.format(str(path1.parent) + '/opentfrecord'))
    os.makedirs(str(path1.parent) + '/opentfrecord', exist_ok=True)

def _distlibution_stratify_(datasetpath, outlstfolder, trainsize, msksetpath):

    # 診断で層別化分割、Testはオリジナル画像のみ
    # Stratified segmentation by diagnosis, test only original image
    toseidatapath = '/workspaces/models/research/deeplab/datasets/pascal_voc_seg/AI_ILD_list_final.csv'
    df_temp = pd.read_csv(toseidatapath, encoding="cp932")
    # Filter existing mask
    exist_mask_file = "/workspaces/models/research/deeplab/datasets/pascal_voc_seg/exist_mask.pkl"
    with open(exist_mask_file, "rb") as handle:
        exist_file_names = pickle.load(handle)
        exist_file_names = exist_file_names.get("exist_files")
    df_temp = df_temp.loc[df_temp['Mask'].isin(exist_file_names)]
    print(len(df_temp))
    df_temp['IPForNot'] = df_temp['diagCategory'].str.contains('IPF')
    df_temp2 = df_temp.loc[df_temp['Mask'].notnull(),['Mask','IPForNot']]
    X = [s.replace('tosei','') for s in df_temp2['Mask'].tolist()]
    y = df_temp2['IPForNot'].tolist()
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=80, stratify=y)


    dcmlist = os.listdir(datasetpath)
    trainlst = [s for s in dcmlist if re.sub('_[0-9]_e[0-9].png', '', s) in X_train]

    testlst = [s for s in dcmlist if re.sub('_[0-9]_e0.png', '', s) in X_val]
    trainvallst = [s for s in dcmlist if re.sub('_[0-9]_e0.png', '', s) in X_train] + testlst

    # train list
    f = open(outlstfolder+'train.txt', 'w', encoding='utf-8')
    for i in trainlst:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    # val list
    f = open(outlstfolder + 'val.txt', 'w', encoding='utf-8')
    for i in testlst:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    # trainval list
    f = open(outlstfolder + 'trainval.txt', 'w', encoding='utf-8')
    for i in trainlst+testlst:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    print('**********************************')
    print('**********************************')
    print('dataset number of {}'.format(outlstfolder))
    df = pd.read_table(outlstfolder + 'train.txt', header=None)
    print('train n = : {}'.format(df.shape[0]))
    df = pd.read_table(outlstfolder + 'val.txt', header=None)
    print('val n = : {}'.format(df.shape[0]))
    df = pd.read_table(outlstfolder + 'trainval.txt', header=None)
    print('trainval n = : {}'.format(df.shape[0]))

    path1 = Path(outlstfolder)
    print('--image_folder= {}'.format(datasetpath))
    print('--semantic_segmentation_folder= {}'.format(str(Path(msksetpath).parent) + '/SegClassRaw'))
    print('--list_folder= {}'.format(outlstfolder))
    print('--output_dir= {}'.format(str(path1.parent) + '/opentfrecord'))
    os.makedirs(str(path1.parent) + '/opentfrecord', exist_ok=True)


def _distlibution_exval_(datasetpath, outlstfolder, trainsize, msksetpath):

    dcmlist = os.listdir(datasetpath)
    lst = [s for s in dcmlist if re.search('[0-9]_[0-9]_e0.png', s)]

    # train list
    f = open(outlstfolder+'train.txt', 'w', encoding='utf-8')
    for i in lst[0]:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    # val list
    f = open(outlstfolder + 'val.txt', 'w', encoding='utf-8')
    for i in lst[0]:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    # trainval list
    f = open(outlstfolder + 'trainval.txt', 'w', encoding='utf-8')
    for i in lst:
        [f.write(s.replace('.png','\n')) for s in dcmlist if i in s]
    f.close()

    print('**********************************')
    print('**********************************')
    print('dataset number of {}'.format(outlstfolder))
    df = pd.read_table(outlstfolder + 'train.txt', header=None)
    print('train n = : {}'.format(df.shape[0]))
    df = pd.read_table(outlstfolder + 'val.txt', header=None)
    print('val n = : {}'.format(df.shape[0]))
    df = pd.read_table(outlstfolder + 'trainval.txt', header=None)
    print('trainval n = : {}'.format(df.shape[0]))

    path1 = Path(outlstfolder)
    print('--image_folder= {}'.format(datasetpath))
    print('--semantic_segmentation_folder= {}'.format(str(Path(msksetpath).parent) + '/SegClassRaw'))
    print('--list_folder= {}'.format(outlstfolder))
    print('--output_dir= {}'.format(str(path1.parent) + '/opentfrecord'))
    os.makedirs(str(path1.parent) + '/opentfrecord', exist_ok=True)



def main():
    start = time.time()


    # # パスの指定
    # # specify the path
    # toseidfpath = '/home/toseidb/df_tosei.csv'
    # efolder = '/home/ToseiKcrc/'

    # toseidcmpath = '/home/toseidb/DCMchoice2/'
    # toseimskpath = '/home/toseidb/baselineMaskNeo/'
    # toseiaugpngpath = efolder + '/tosei/png/'
    # toseiaugmskpath = str(Path(toseiaugpngpath).parent) + '/mskgry/'
    # toseilst = str(Path(toseiaugpngpath).parent) + '/lst/'

    # kcrcdcmpath = '/home/kcrcdb/dcmchoice/'
    # kcrcmskpath = '/home/kcrcdb/maskpng/'
    # kcrcaugpngpath = efolder + '/kcrc/png/'
    # kcrcaugmskpath = str(Path(kcrcaugpngpath).parent) + '/mskgry/'
    # kcrclst = str(Path(kcrcaugpngpath).parent) + '/lst/'


    # opendcmpath = '/home/covid19open/tif/'
    # openmskpath = '/home/covid19open/mask/'
    # openaugpngpath = efolder + '/open/openpng/'
    # openaugmskpath = str(Path(openaugpngpath).parent) + '/openmskind/'
    # openlst = str(Path(openaugpngpath).parent) + '/openlst/'



    # # # *******************************
    # # # *******************************
    # # shutil.rmtree(openaugpngpath)
    # # shutil.rmtree(openaugmskpath)
    # # # *******************************
    # # # *******************************



    # # 実験用フォルダ作成 python 3.2以降は例外規定を書かなくてOK
    # # Create a folder for experiments You don't have to write exceptions for python 3.2 or later
    # os.makedirs(toseiaugpngpath, exist_ok=True)
    # os.makedirs(toseiaugmskpath, exist_ok=True)
    # os.makedirs(kcrcaugpngpath, exist_ok=True)
    # os.makedirs(kcrcaugmskpath, exist_ok=True)
    # os.makedirs(openaugpngpath, exist_ok=True)
    # os.makedirs(openaugmskpath, exist_ok=True)
    # os.makedirs(toseilst, exist_ok=True)
    # os.makedirs(kcrclst, exist_ok=True)
    # os.makedirs(openlst, exist_ok=True)

    datasetname = 'IPF' # IPF VOC2012
    root_dir = '/workspaces/models/research/deeplab/'
    efolder = root_dir + 'datasets/pascal_voc_seg/{}/generated/'.format(datasetname)

    toseidcmpath = root_dir + 'datasets/pascal_voc_seg/VOC2012/original/dcmchoice/'
    toseimskpath = root_dir + 'datasets/pascal_voc_seg/VOC2012/original/mask/'
    toseiaugpngpath = efolder + 'PNGImages/'
    toseiaugmskpath = efolder + 'SegmentationClass/'
    toseilst = efolder + 'ImageSets/Segmentation/'

    os.makedirs(toseiaugpngpath, exist_ok=True)
    os.makedirs(toseiaugmskpath, exist_ok=True)
    os.makedirs(toseilst, exist_ok=True)

    # dcm to pngが必要な場合、コマンドラインで'dcm_to_png'を指定
    # If you need dcm to png, specify 'dcm_to_png' on the command line
    if (sys.argv[1] == 'dcm_to_png') or (len(os.listdir(toseiaugpngpath)) < 10):
        _DCMtoPNG_(
            dcmfolder = toseidcmpath,
            mskfolder = toseimskpath,
            outputpngfolder = toseiaugpngpath,
            outputmskfolder = toseiaugmskpath
        )

        _distlibution_stratify_(
            datasetpath = toseiaugpngpath,
            msksetpath = toseiaugmskpath,
            outlstfolder = toseilst,
            trainsize = 0.9)



        # _DCMtoPNG_(
        #     dcmfolder = kcrcdcmpath, 
        #     mskfolder = kcrcmskpath, 
        #     outputpngfolder = kcrcaugpngpath, 
        #     outputmskfolder = kcrcaugmskpath
        # )

        # _distlibution_exval_(
        #     datasetpath = kcrcaugpngpath,
        #     msksetpath = kcrcaugmskpath,
        #     outlstfolder = kcrclst, 
        #     trainsize = 0.99)


        # ********************************
        # ********************************
        # Open Source Lung
        # _TiftoPNG_(
            # dcmfolder = opendcmpath, 
            # mskfolder = openmskpath, 
            # outputpngfolder = openaugpngpath, 
            # outputmskfolder = openaugmskpath
        # )

        # # convert index color to グレースケール画像
        # _IndexGray_(
            # inputmskfolder = openaugmskpath
        # )

        # _distlibution_openlst_(
            # datasetpath = openaugpngpath,
            # msksetpath = openaugmskpath,
            # outlstfolder = openlst, 
            # trainsize = 0.8)






    # *************************
        # pretraining for opendataset
    # *************************
    # train n = : 3332
    # val n = : 848
    # trainval n = : 4180

    # --image_folder= /home/ToseiKcrc/open/openpng/
    # --semantic_segmentation_folder= /home/ToseiKcrc/open/SegClassRaw/
    # --list_folder= /home/ToseiKcrc/open/openlst/
    # --output_dir= /home/ToseiKcrc/open/opentfrecord



    # dataset number of /home/ToseiKcrc/kcrc/lst/
    # train n = : 15216
    # val n = : 976
    # trainval n = : 16192
    # --image_folder= /home/ToseiKcrc//tosei/png/
    # --semantic_segmentation_folder= /home/ToseiKcrc/tosei/SegClassRaw
    # --list_folder= /home/ToseiKcrc/tosei/lst/
    # --output_dir= /home/ToseiKcrc/tosei/opentfrecord



    # vi /home/models/research/deeplab/datasets/data_generator.py

    # "i" ; insert mode
    # change numbers
    # "Escape"
    # ":wq""Enter"


    # # 下記実行後、tfrecordディレクトリにTFRecordデータセットが出力される。
    # # After executing the following, a TFRecord dataset will be output to the tfrecord directory.
    # python /home/models/research/deeplab/datasets/build_voc2012_data.py --image_folder="/home/ToseiKcrc/kcrc/png" --semantic_segmentation_folder="/home/ToseiKcrc/kcrc/mskgry" --list_folder="/home/ToseiKcrc/kcrc/lst" --image_format="png" --output_dir="/home/ToseiKcrc/kcrc/opentfrecord"


    # # 学習の実行
    # # Run training
    # cd /home/models/research/deeplab

    # ********************
    # 初期 (Initial)
    # ********************
    # python train.py  --logtostderr=true --training_number_of_steps=10000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size="513,513" --train_batch_size=4 --dataset="pascal_voc_seg" --tf_initial_checkpoint="./datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt" --train_logdir="/home/ToseiKcrc/open/opentrainlog" --dataset_dir="/home/ToseiKcrc/open/opentfrecord" --fine_tune_batch_norm=false --initialize_last_layer=true --last_layers_contain_logits_only=false

    # ********************
    # Transfer
    # ********************
    # python train.py  --logtostderr=true --training_number_of_steps=50000 --train_split="train" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --train_crop_size="513,513" --train_batch_size=4 --dataset="pascal_voc_seg" --tf_initial_checkpoint="/home/ToseiKcrc/open/model.ckpt-10000" --train_logdir="/home/ToseiKcrc/tosei/trainlog" --dataset_dir="/home/ToseiKcrc/tosei/opentfrecord" --fine_tune_batch_norm=false --initialize_last_layer=true --last_layers_contain_logits_only=false


    # # 引数名	意味	値（例）(Argument name Meaning Value (example))
    # # logtostderr	ログ出力の有無	－ (Presence or absence of log output)
    # # training_number_of_steps	学習回数	30000
    # # train_split	使用データ	train, val, trailval
    # # model_variant	識別モデル種類	xception_65, mobilenet_v2
    # # atrous_rates	Atrous畳み込みの比率 ※複数回設定可能	6
    # # output_stride	出力ストライド（atrous_rateとの組み合わせ）	16
    # # decoder_output_stride	入出力の空間解像度の比率	4
    # # train_crop_size	画像の切り出しサイズ ※XYの2回指定	513
    # # train_batch_size	ミニバッチのサイズ	64
    # # fine_tune_batch_norm	Batch Normalizationの実行	true, false ※GPUで学習するときはfalse
    # # dataset	データセット名	cityscapes, pascal_voc_seg, ade20k
    # # tf_initial_checkpoint	学習済みモデル名	deeplab/datasets/pascal_voc_seg/init_models/deeplabv3_pascal_train_aug/model.ckpt
    # # initialize_last_layer	最後のレイヤーの初期化	true, false # クラス数を変えたときはfalse
    # # last_layers_contain_logits_only	logitsを最後のレイヤーとしてのみ考慮	true, false # クラス数を変えたときはtrue
    # # train_logdir	ログ出力フォルダ名	deeplab/datasets/pascal_voc_seg/exp/train_on_trainval_set/train
    # # dataset_dir	データセットフォルダ名	deeplab/datasets/pascal_voc_seg/tfrecord


    # # tensorboardで学習進捗の確認
    # "ctrl" P Q

    # # # Windowsマシンからポートフォワード
    # # ssh -L 8081:localhost:8888 user@server.host.name

    # # WindowsマシンからAnaconda Prompt立ち上げ
    # tensorboard --logdir=X:/ToseiKcrc/tosei/trainlog

    # # ブラウザから、
    # # http://hogehogePC:6060
    # # にアクセス


    # # 学習済みデータを使用して評価
    # cd /home/models/research/deeplab
    # python eval.py --logtostderr --eval_split="trainval" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --eval_crop_size="513,513" --max_resize_value=512 --min_resize_value=512 --checkpoint_dir="/home/ToseiKcrc/tosei/trainlog" --eval_logdir="/home/ToseiKcrc/kcrc/evallog" --dataset_dir="/home/ToseiKcrc/kcrc/opentfrecord" --max_number_of_evaluations=1 --eval_interval_secs=0 --dataset="pascal_voc_seg"




# 25k
# eval/miou_1.0_class_0[0.987986863]
# eval/miou_1.0_class_1[0.833726346]
# eval/miou_1.0_class_4[0.31147933]
# eval/miou_1.0_class_9[0.404807717]
# eval/miou_1.0_overall[0.634500086]


    # tosei
    # 3万 (30,000)
    # eval/miou_1.0_class_0[0.987382114]
    # eval/miou_1.0_class_1[0.821584046]
    # eval/miou_1.0_class_4[0.339685082]
    # eval/miou_1.0_class_9[0.414949775]
    # eval/miou_1.0_overall[0.640900314]

    # ＋６万 (+60,000)
    # eval/miou_1.0_class_0[0.987636507]
    # eval/miou_1.0_class_1[0.825396657]
    # eval/miou_1.0_class_4[0.355198175]
    # eval/miou_1.0_class_9[0.439754605]
    # eval/miou_1.0_overall[0.651996434]

    # +12万  (+120,000)
    # eval/miou_1.0_class_0[0.987941086]
    # eval/miou_1.0_class_1[0.829354465]
    # eval/miou_1.0_class_4[0.379993975]
    # eval/miou_1.0_class_9[0.467240274]
    # eval/miou_1.0_overall[0.66613239]

    # # +24万 (+240,000)
    # eval/miou_1.0_class_0[0.988202]
    # eval/miou_1.0_class_1[0.83458662]
    # eval/miou_1.0_class_4[0.413078874]
    # eval/miou_1.0_class_9[0.48651]
    # eval/miou_1.0_overall[0.680594385]

    # kcrc
    # eval/miou_1.0_class_0[0.938273191]
    # eval/miou_1.0_class_1[0.633554161]
    # eval/miou_1.0_class_4[0.259148628]
    # eval/miou_1.0_class_9[0.0567857921]
    # eval/miou_1.0_overall[0.471940398]


    # # 可視化 (Visualization)
    # cd /home/models/research/deeplab
    # python vis.py --logtostderr --vis_split="trainval" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size="513,513" --max_resize_value=512 --min_resize_value=512 --checkpoint_dir="/home/ToseiKcrc/tosei/trainlog" --vis_logdir="/home/ToseiKcrc/tosei/vislog" --dataset_dir="/home/ToseiKcrc/tosei/opentfrecord" --max_number_of_iterations=1 --eval_interval_secs=0 --dataset="pascal_voc_seg"

    # python vis.py --logtostderr --vis_split="trainval" --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4 --vis_crop_size="513,513" --max_resize_value=512 --min_resize_value=512 --checkpoint_dir="/home/ToseiKcrc/tosei/trainlog" --vis_logdir="/home/ToseiKcrc/kcrc/vislog" --dataset_dir="/home/ToseiKcrc/kcrc/opentfrecord" --max_number_of_iterations=1 --eval_interval_secs=0 --dataset="pascal_voc_seg"


    # cd /home/models/research/deeplab
    # python export_model.py --checkpoint_path="/home/ToseiKcrc/open/opentrainlog/model.ckpt-10000" --export_path="/home/ToseiKcrc/open/frozen_inference_graph.pb" --num_classes=21 --model_variant="xception_65" --atrous_rates=6 --atrous_rates=12 --atrous_rates=18 --output_stride=16 --decoder_output_stride=4


    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

if __name__ == "__main__":
    main()
    # img_path = 'data/CT_Original'
    # list_img = ['101_1']
    # for ind, filename in enumerate(list_img):      
    #     dicom_img_path = os.path.join(img_path, filename)
    #     if not os.path.isfile(dicom_img_path):
    #         print("{} is not a file.".format(dicom_img_path))
    #         continue  
    #     xITK = sitk.ReadImage(dicom_img_path)
    #     x0 = sitk.GetArrayFromImage(xITK)
    #     x = np.float32(np.squeeze(x0))
    #     print("Shape: {} - {}".format(x0.shape, type(x0)))        
    #     cv2.imshow("original dicom image {}".format(dicom_img_path), x)

    #     converted_x = _conv_dcm_(dcmfullpath=dicom_img_path)
    #     print("Shape: {} - {}".format(converted_x.shape, type(converted_x))) 
    #     cv2.imshow("converted dicom image {}".format(dicom_img_path), converted_x)
    #     cv2.waitKey(0)

    # img_path = 'data/Lung_Mask/101_1.png'
    # img_data = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
    # print("Shape: {}".format(img_data.shape))
    # # print(img_data[img_data > 0])
    # cv2.imshow("Masked", img_data)    
    # cv2.waitKey(0)

    # test_path = '/workspaces/models/research/deeplab/datasets/pascal_voc_seg/VOC2012/generated/SegmentationClassRaw'
    # labels_list = list()
    # for image_name in os.listdir(test_path):
    #     img_path = os.path.join(test_path, image_name)
    #     data_img = np.array(Image.open(img_path))
    #     print("Shape: {}".format(data_img.shape))
    #     labels_list.extend(list(set(data_img.flatten())))

    # print("xxxxxxxxxxxxxxxxxxxxxxxx labels_list: {}".format(list(set(labels_list))))
    



# docker run -it --gpus all --mount type=bind,src=/data2/.,dst=/home -p 8081:8888 rapidsai/rapidsai:latest_imp
# docker build -t tf/dlv3plus:1.0 <Dockerfile path>
# # 「-t」オプションは作成するDockerイメージのイメージ名およびタグ名を指定します。
# # The "-t" option specifies the image name and tag name of the Docker image to create.
