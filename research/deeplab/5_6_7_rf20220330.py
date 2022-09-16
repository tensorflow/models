# -*- coding: utf-8 -*-
# segmentationされた結果画像が保存されている時からRFするスクリプト
# Script to RF from when the segmented result image is saved
import time
import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np
# from tkinter import filedialog
import os  # needed navigate the system to get the input data
import six
import sys
import re
# import matplotlib.pyplot as plt
import itertools
from openpyxl import load_workbook
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
import cv2
import collections
from PIL import Image
from operator import itemgetter
# from sklearn.preprocessing import StandardScaler 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier#ランダムフォレスト


import cudf
from cuml import RandomForestClassifier as cuRF
from cuml import accuracy_score as accuracy_score
import cupy as cp


def flatten(l):
    # '可変ネストされたリストを平坦化
    # 'flatten variable nested list
    for el in l:
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def _loadxlsx_(filepath,sheetname):
    # xlsx fileの読み込み
    wb = load_workbook(filename = filepath)
    ws = wb[sheetname]
    data = ws.values
    data = list(data)
    return pd.DataFrame(data[1:], columns=data[0])


def _readtxt_(path):
    f = open(path, 'r')
    # 形式を変更する(filename/nfilename/n...)
    l_lst = f.read().splitlines()
    f.close()
    return(l_lst)

def _linear_(df,X,Y):
    # % ロジスティク回帰
    # % logistic regression
    dfcomp = df[[X,Y]].dropna(how='any')
    lr = LinearRegression()
    my_lr = lr.fit(dfcomp[X].values.reshape(-1, 1), dfcomp[Y].values.reshape(-1, 1))
    #検証用データで予測を求める
    # get predictions with validation data
    df[(df[Y].isnull()) * (~df[X].isnull())][X] = my_lr.predict(
        df[(df[Y].isnull()) * (~df[X].isnull())][X].values.reshape(-1, 1)
        )
    return df


def _elo_shr_(f_cimg, f_w, f_h, f_shrate, xc, yc):
    # 円の拡大縮小して外接する楕円を作成　5分割
    # Create an ellipse that circumscribes the circle by scaling it divided into 5
    imgzero = np.zeros([512,512])
    img2 = cv2.resize(f_cimg, dsize=(int(f_w*f_shrate), int(f_h*f_shrate)))
    hr = int(img2.shape[0] / 2) # hight
    wr = int(img2.shape[1] / 2) # width

    # np.zeros([512,512]) + img2
    imgzero[yc-hr:yc-hr+img2.shape[0], xc-wr:xc-wr+img2.shape[1]] = img2
    imgzero[imgzero>0] = 1 # 計算するために01に変更 (change to 01 to calculate)
    return(imgzero)

def _elo_shr2_(f_cimg, f_w, f_h, f_shrate, xc, yc):
    # 円の拡大縮小して外接する楕円を作成　5分割
    # Create an ellipse that circumscribes the circle by scaling it divided into 5

    # 領域境界を2ピクセルカウントしない
    # don't count region boundaries by 2 pixels
    # k=2
    k=0

    imgzero = np.zeros([512,512])
    img2 = cv2.resize(f_cimg, dsize=(int(f_w*f_shrate)-k, int(f_h*f_shrate)-k)) #dsize=(width, height)
    hr = int(img2.shape[0] / 2) # hight
    wr = int(img2.shape[1] / 2) # width

    imgzero[yc-hr:yc-hr+img2.shape[0], xc-wr:xc-wr+img2.shape[1]] = img2
    imgzero[imgzero>0] = 1 # 計算するために01に変更 (# change to 01 to calculate)
    return(imgzero)

def _imgcount_(r, grayimg,imgdcm):
    # 要素数をカウント gray scale時のラベル [0; background, 22; IPF, 57; non-IPF, 15; lung]
    # Count the number of elements Label in gray scale [0; background, 22; IPF, 57; non-IPF, 15; lung]
    npcolor = [15, 57]
    nplung = [255]
    temp = [np.count_nonzero(grayimg*r == s) for s in npcolor] + [np.count_nonzero(imgdcm*r == nplung)]

    return(temp)


def _ctcal_(fname,imgfolder):
    # imgPIL = Image.open(imgfolder + fname)  # DL結果画像読み込み (DL result image loading)
    # arrPIL = np.asarray(imgPIL) # (512, 512, 3)
    # grayimg = cv2.cvtColor(arrPIL, cv2.COLOR_BGR2GRAY) # モノクロへ (to monochrome)

    grayimg = np.array(
                Image.open(imgfolder + fname).convert('L')
            )

    # IPF, NonIPFを検出するアルゴリズムにした場合、DL画像に肺は出ないので、一緒にvislogで一緒に産出される画像から中心座標を算出する
    # When using an algorithm that detects IPF and NonIPF, the lungs do not appear in the DL image, so calculate 
    # the center coordinates from the images produced together with vislog
    imgdcm = np.array(
                Image.open(imgfolder + fname.replace('_prediction.png','_image.png')).convert('L')
            )
    imgdcm[imgdcm > 0] = 255

    x, y, w, h = cv2.boundingRect(imgdcm) # 輪郭の検出 (contour detection)

    # 中心座標の算出 (Calculation of center coordinates)
    xc = int(x + w/2)
    yc = int(y + h/2)
    # 円を作成しておく (create a circle)
    cimg = cv2.circle(np.zeros([512,512]), (256, 256), 256, (255, 255, 255), thickness=-1)

    # 分割するベースの領域を策定 (Formulate the area of the base to divide)
    r1 = _elo_shr2_(cimg,w,h,1/5, xc, yc)
    r2 = _elo_shr2_(cimg,w,h,2/5, xc, yc) - _elo_shr_(cimg,w,h,1/5, xc, yc)
    r3 = _elo_shr2_(cimg,w,h,3/5, xc, yc) - _elo_shr_(cimg,w,h,2/5, xc, yc)
    r4 = _elo_shr2_(cimg,w,h,4/5, xc, yc) - _elo_shr_(cimg,w,h,3/5, xc, yc)
    r5 = 1 - _elo_shr_(cimg,w,h,4/5,xc, yc)

    # 要素数をカウント gray scale時のラベル [0; background, 22; IPF, 38; non-IPF, 128; lung]
    # Count the number of elements Label in gray scale [0; background, 22; IPF, 38; non-IPF, 128; lung]
    ctcal_ret = _imgcount_(r1, grayimg, imgdcm) + _imgcount_(r2, grayimg, imgdcm) + _imgcount_(r3, grayimg, imgdcm) + _imgcount_(r4, grayimg, imgdcm) + _imgcount_(r5, grayimg, imgdcm)

    return(ctcal_ret)


def _slicesum_(arr_CTperPts, locationlist, arr_dis, m_val, baseloc):
    # 症例ごとのラベルの合計を並べた配列を、それぞれ計算する
    # Calculate an array that lists the total number of labels for each case
    # ' (ctcal_bycase, l_ipfloc, l_i, m, addval)
    x = np.sum(
        arr_CTperPts[:,
            [s + baseloc for s in locationlist[arr_dis[m_val][0]:arr_dis[m_val][-1]+1]]
        ].astype('uint32'), axis=1
        )
    x = x + np.ones(x.shape).astype('uint32')
    return(x)


def _pd_np_forML_(lst,df,colname,hspname):
    # Training、Valのリストから症例を抽出して元のデータシートにくっつける
    # Extract cases from Training and Val lists and attach them to the original datasheet
    lst_temp = set([hspname + s for s in lst])
    df_temp = pd.DataFrame(lst_temp)
    pdX = pd.merge(df_temp, df, left_on=0, right_on='Mask', how='inner')
    pdX = pdX.dropna(subset=['IPForNot'])
    pdY = pdX['IPForNot'].astype('float64')
    pdX = pdX[colname]
    pdX = pdX.drop('Mask', axis=1)
    return(pdX, pdY)


def _connect_CTresult_(v_ctcal_bycase,v_l_i,v_m,v_addval):
    # ４スライスを１スライスに結合する
    # 症例 * (IPF or nonIPF or lung) のnp.arrayを作成
    # combine 4 slices into 1 slice
    # Create np.array for case * (IPF or nonIPF or lung)
    IPF_sum = _slicesum_(v_ctcal_bycase, l_ipfloc, v_l_i, v_m, v_addval)
    nonIPF_sum = _slicesum_(v_ctcal_bycase, l_nonipfloc, v_l_i, v_m, v_addval)
    lung_sum = _slicesum_(v_ctcal_bycase, l_lungloc, v_l_i, v_m, v_addval)
    # 横に結合する
    # join horizontally
    v_ctcalsum_per_slice = np.vstack(
        [IPF_sum / lung_sum, 
        nonIPF_sum / lung_sum, 
        IPF_sum/(IPF_sum + nonIPF_sum)
    ]).T
    v_ctcalsum_per_slice = np.nan_to_num(v_ctcalsum_per_slice) # fill nan
    return(v_ctcalsum_per_slice)


def _cudf_from_pd_(df):
    # 'convert float64 to float32
    # df_cp=cp.asarray(np.array(df))
    df2 = pd.DataFrame(np.array(df))
    df_cp = cudf.from_pandas(df2)
    df_cp32 = df_cp.astype('float32')
    return(df_cp32)


def _rf_dfmake_(dfX,dfy):
    dfXg = _cudf_from_pd_(
        dfX.dropna(how='any').values
        )
    dfyg = _cudf_from_pd_(
        dfy[~dfX.isnull().any(axis=1)]
        )

    return dfXg, dfyg


def _rftrain_(df_X,df_y):
    # 'Random Forest (cuRF)
        # 'n_streams; int (default = 4)
            # 'Number of parallel streams used for forest building.
        # 'min_samples_leaf: int or float (default = 1)
        # 'min_samples_split: int or float (default = 2)
    df_X_g, df_y_g = _rf_dfmake_(df_X,df_y)

    cu_rf = cuRF(
        n_estimators = 350, 
        max_depth = 8, 
        n_bins = 8, 
        n_streams = 4,
        min_samples_leaf = 8,
        random_state = None,
        verbose = 0
        )
    # min_samples_split = 4,
    # min_samples_leaf -> 2,4,8
    # n_jobs = -1
    # 75%, 230s of 512bins

    cu_rf.fit(df_X_g, df_y_g)

    # トレーニングデータに対する精度 OOBはない! GPUでpredictすると、float32しか受け付けないので注意
    # No precision OOB for training data! Note that predicting on GPU only accepts float32
    accuracdf_y = accuracy_score(
        cu_rf.predict(df_X_g,predict_model='GPU'),
        df_y_g
        )
    return (accuracdf_y, cu_rf)


def _rftest_(df_X,df_y,cu_rf):
    df_X_g, df_y_g = _rf_dfmake_(df_X,df_y)
    # トレーニングデータに対する精度 OOBはない! GPUでpredictすると、float32しか受け付けないので注意
    # No precision OOB for training data! Note that predicting on GPU only accepts float32
    accuracdf_y = accuracy_score(
        cu_rf.predict(df_X_g,predict_model='GPU'),
        df_y_g
        )
    return (accuracdf_y)


def _readdf_(path):
    df_temp = pd.read_csv(path,encoding='cp932')
    df_temp = df_temp[df_temp['target']==1]

    # pythonで使えない変数名があるので置換する
    # Since there is a variable name that cannot be used in python, replace it
    df_temp = df_temp.rename(columns={'Mφ': 'Mac', 'CD4/CD8': 'CD4CD8', 'FEV1_FVC': 'FEV1FVC'})

    # % IPForNotを再計算
    # Recalculate % IPForNot
    df_temp['IPForNot'] = df_temp['diagnosis'].str.contains('IPF')
    return df_temp



def _calcCT_(folderpath):
    # 対象画像のリスト
    # list of target images
    imgfolder = folderpath + 'vislog/segmentation_results/'
    imgfiles = os.listdir(imgfolder)

    # **_predictionがラベル画像 (n=3240)
    # **_prediction is label image (n=3240)
    l_predname = [s for s in imgfiles if '_prediction' in s]
    # ['000000_prediction.png', '000001_prediction.png', '000002_prediction.png']
    
    # lstからファイル名を読み込む train & val一括
    # read file name from lst train & val all at once
    l_trainval = _readtxt_(folderpath+'lst/trainval.txt')
    # オーグメンテーションしていないファイルの位置を取得
    # get position of non-augmented file
    l_orgfileloc = [l_trainval.index(s) for s in l_trainval if re.match('^[0-9]+_[0-9]+_e0', s.replace('KCRC',''))]

    l_ctcal = []
    for rowi in l_orgfileloc:
        # 画像読んで、5箇所に分割
        # Read the image and divide it into 5 parts
        fname = l_predname[rowi]
        l_ctcal.append(_ctcal_(fname,imgfolder) + [l_trainval[rowi]])
    # [[0, 0, 0, 0, 0, 1559, 0, 0, 6647, 0, 0, 12990, 0, 1573, 22605, '101_2'],
    # [0, 0, 0, 0, 0, 947, 0, 0, 2202, 0, 874, 4389, 0, 1733, 18689, '347_4'], 

    # listからnp.arrayに変換
    # Convert from list to np.array
    ctcal = np.array(l_ctcal).reshape(-1,3*5+1)

    ctcal = ctcal[np.argsort(ctcal[:,-1])] # ファイル名順にする (Sort by file name)
    ctcal_bycase = ctcal.reshape(-1,(ctcal.shape[1])*4) # 症例毎の行にする (Make a row for each case)
    return ctcal_bycase


def _ctcalcsum_(ctcal,slice_num,l_,arr):
    nptemp = ctcal[:,3*5*slice_num:3*5*(slice_num+1)]
    for dis_i in l_:
        # lungの位置から、分割して統合した場合の、各々の合計を算出し割合を計算。
        # From the lung position, calculate the total and calculate the ratio when dividing and integrating
        lung_sum = np.sum(nptemp[:,dis_i*3 -1],axis=1) +1
        ipf_sum = np.sum(nptemp[:,dis_i*3 -3],axis=1) +1
        nonipf_sum = np.sum(nptemp[:,dis_i*3 -2],axis=1) +1

        # 統合毎にデータが３つできる
        # 3 data are generated for each integration
        T = np.vstack([ipf_sum / lung_sum, nonipf_sum / lung_sum, ipf_sum / nonipf_sum])
        arr = np.append(arr, T, axis=0) # 空配列に足していく (add to empty array)
    return arr


def _df_l_(arr,hspname,ctcal,l_,df_,l_temp):
    # ctcal >> '805_4_e0'


    arr = arr.transpose()
    arr[np.isnan(arr)] = 0

    cname = np.array(
            [hspname + re.match('^[0-9]+',s.replace('KCRC','')).group() for s in ctcal[:,-1].tolist()]
        )
    # KCRCは画像にKCRCと入っているので、それを除去してcname作成
    # KCRC is in the image with KCRC, remove it and create cname

    coltemp = list(itertools.chain.from_iterable(l_)) + ['Mask']
    df_ctresult = pd.DataFrame(np.append(arr,cname.reshape(cname.shape[0],-1),axis=1))
    df_ctresult.columns = coltemp

    # データ結合
    # data binding
    df_all = pd.merge(df_, df_ctresult, how='inner', left_on='Mask',right_on='Mask')

    df_fill = df_all.dropna(subset=['IPForNot'])
    # 読み込む変数を指定
    # specify variables to read
    l_scale = list(flatten(l_temp + coltemp)) # ネストしたlistを使える形に戻す (Make nested list usable)
    # t_scale = [tuple(s) for s in l_scale if s not in ['Mask']]
    # l_out = [list(s) for s in list(set(t_scale))]
    l_out = list(dict.fromkeys(l_scale))


    return(l_out, df_fill)


def main():
    start = time.time()
    print('********************')
    print('  -----  program run  -----  ')
    print('********************')

    efolder = '/home/ToseiKcrc/'
    
    # 画像処理したフォルダ (# image processing folder)
    orgfolder = efolder + 'tosei/'
    testfolder = efolder + 'kcrc/'

    # 臨床データを読み込む (# load clinical data)
    ctcal_path = orgfolder + 'ctcal_del.npy' # ct DL結果を分割した結果を格納 (Stores the result of splitting the DL result)
    df_tosei_path = orgfolder+ 'df_tosei_del.csv' # 臨床情報の欠損値処理をした結果を格納 (Stores the results of missing value treatment of clinical information)
    # 臨床データを読み込む (# Load clinical data)
    ctcal_testpath = testfolder + 'ctcal_del.npy' # ct DL結果を分割した結果を格納 (Stores the result of splitting the DL result)
    df_test_path = testfolder+ 'df_test_del.csv' # 臨床情報の欠損値処理をした結果を格納 (Stores the results of missing value treatment of clinical information)


    # % compensent min value
    mincomplist = [
        'PackYear','ANA','antiARS_Ab','RA','antiCCP','aldolase',
        'antiRNPAb','antiSCL70','antiRNApoly3','centromere',
        'antiSSA_Ab','antiSSB_Ab',
        ]
    # 最頻値で埋める (fill with mode)
    l_mode = ['IgG', 'IgA', 'IgM', 'ANA', 'KL6', 'SPD',
            'CentromereType', 'GranularType', 'HomogeneousType', 'NucleolarType',
            'PeripheralType', 'SpeckledType', 'NuclearEnvelopeType', 'CytoplasmicType',
            'antiARS_Ab', 'MPOANCA', 'CANCA', 'RA', 'antiCCP', 'Jo1', 'aldolase', 
            'antiRNPAb', 'antiSCL70', 'antiSm', 'antiRNApoly3', 'centromere', 
            'antiSSA_Ab', 'antiSSB_Ab', 'antiDNA_Ab', 'antiDsDNA'
            ]
    # 最近傍で欠損値を埋める (Fill Missing Values with Nearest Neighbors)           # 並び替えてからやるように！ (Rearrange and do it!)
    l_pca = ['TCC', 'Neu', 'Lym', 'Eo', 'Mac', 'CD4CD8']
    # ANAは処理する (NAN handle)
    analist = [
        'CentromereType', 'GranularType', 'HomogeneousType', 'NucleolarType',
        'PeripheralType', 'SpeckledType', 'NuclearEnvelopeType', 'CytoplasmicType'
    ]
    tlist = ['age','BMI','PackYear',
             'TCC','Neu','Lym','Mac','Eo','CD4CD8',
             'perFVC','perDLCO','FEV1FVC',
             'IgG','IgA','IgM',
             'ANA','antiARS_Ab','MPOANCA','CANCA',
             'RA','antiCCP','Jo1','aldolase','antiRNPAb','antiSCL70',
             'antiSm','antiRNApoly3','centromere','antiSSA_Ab','antiSSB_Ab','antiDNA_Ab','antiDsDNA',
             'KL6','SPD'
            ]
    # 'M_','CD4_CD8','FEV1_FVC',



    # CT分割計算結果とデータフレームを格納する (Store CT split calculation results and data frames)
    
    # データフレームから (from the dataframe)
    if os.path.exists(df_tosei_path) and os.path.exists(df_test_path):
        df_tosei = pd.read_csv(df_tosei_path)
        df_test = pd.read_csv(df_test_path)

    else:
        # 臨床データ読み込み (Clinical data loading)
        df_tosei = _readdf_(orgfolder + 'AI_ILD_list_final.csv')
        df_test = _readdf_(testfolder + 'for_rf_datasheet.csv')


        # % ***************************
        # % 2群比較して、有意だったものはそのまま採用する (When comparing the 2 groups, those that are significant are adopted as they are.)
        # % ***************************

        pcut = 0.2
        # 等分散性の仮定は一回無視 (Ignore the homoscedasticity assumption once)
        df_tosei_ipf = df_tosei[df_tosei['IPForNot'] == 1]
        df_tosei_nonipf = df_tosei[df_tosei['IPForNot'] == 0]

        l_result = []
        for i in range(len(tlist)):
            t, p = stats.ttest_ind(
                df_tosei_ipf[tlist[i]].dropna(how='all'), df_tosei_nonipf[tlist[i]].dropna(how='all'),
                equal_var = False)
            if p < pcut:
                l_result.append(tlist[i])
            # print("t値：{0}, p値：{1}".format(t, p))


        xlist = ['gender',
                 'CentromereType','GranularType','HomogeneousType','NucleolarType',
                 'PeripheralType','SpeckledType','NuclearEnvelopeType','CytoplasmicType'
                ]
        for i in range(len(xlist)):
            crossed = pd.crosstab(
                df_tosei['IPForNot'], df_tosei[xlist[i]]
                )
            x2, p, dof, expected = stats.chi2_contingency(crossed)  
            if p < pcut:
                l_result.append(tlist[i])


        # % ================================
        # % 欠損値補完 (missing value imputation)
        # % ================================

        # % pulmonary function test -> 全欠損が4例/1066例　FEV1 6例、 DLco 39例欠損 (4 cases/1066 total defects, 6 FEV1 cases, 39 DLco defects)
        # %  > NMARなので、線形補完する? (NMAR, so linear interpolation)
        df_tosei = _linear_(df_tosei,'perFVC','perDLCO')
        df_tosei = _linear_(df_tosei,'perFVC','FEV1FVC')
        df_test = _linear_(df_test,'perFVC','perDLCO')
        df_test = _linear_(df_test,'perFVC','FEV1FVC')


        # % NaNは削除
        df_tosei = df_tosei.dropna(subset=['age','BMI','gender','perFVC','FEV1FVC','perDLCO'])
        df_test = df_test.dropna(subset=['age','BMI','gender','perFVC','FEV1FVC','perDLCO'])


        df_tosei[mincomplist] = df_tosei[mincomplist].fillna(df_tosei[mincomplist].min())
        df_test[mincomplist] = df_test[mincomplist].fillna(df_tosei[mincomplist].min())


        imputer = KNNImputer(n_neighbors=5)
        df_tosei[l_pca] = pd.DataFrame(imputer.fit_transform(df_tosei[l_pca]))
        df_test[l_pca] = pd.DataFrame(imputer.transform(df_test[l_pca]))


        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        df_tosei[l_mode] = pd.DataFrame(imputer.fit_transform(df_tosei[l_mode]))
        df_test[l_mode] = pd.DataFrame(imputer.transform(df_test[l_mode]))


        f_trans01 = lambda x: 1 if x > 0 else 0
        df_tosei[analist] = df_tosei[analist].applymap(f_trans01)
        df_test[analist] = df_test[analist].applymap(f_trans01)
        for i in analist:
            df_tosei[i] = df_tosei[i] * df_tosei['ANA']
            df_test[i] = df_test[i] * df_test['ANA']

        df_tosei.to_csv(df_tosei_path, index=False)
        df_test.to_csv(df_test_path, index=False)



    # ******************************
    # segmentationした画像の加工を行う (Process the segmented image)
    # ******************************
    if os.path.exists(ctcal_path):
        ctcal_bycase = np.load(ctcal_path)
    else:
        ctcal_bycase = _calcCT_(orgfolder)
        np.save(ctcal_path, ctcal_bycase)

    if os.path.exists(ctcal_testpath):
        ctcal_testbycase = np.load(ctcal_testpath)
    else:
        ctcal_testbycase = _calcCT_(testfolder)
        np.save(ctcal_testpath, ctcal_testbycase)




    # 区切り文字を入れる位置で,CT分割方法の組み合わせを算出
    # (Calculate the combination of CT division methods at the position where the delimiter is inserted)
    l_temp = [list(itertools.combinations(list(range(1,5)),s)) for s in range(1,6)]
    comb = list(itertools.chain.from_iterable(l_temp))
    l_15 = list(range(1,6))
    l_dis = [(np.array_split(l_15, comb[s])) for s in range(len(comb))]
    # l_dis = [array([1, 2, 3, 4]), array([5])],
    #         [array([1]), array([2]), array([3, 4, 5])],...

    # 1:15 の４つの組み合わせ (four combinations of)
    x = range(15)
    combvec = list(itertools.product(x, x, x, x))


    l_svmresult = []

    # *********************
    # 最初に行ったランダムスプリットのRF結果を読み込んで、結果が良かった分割方法のみためす 
    # (Read the RF results of the first random split and try only the split method that gave good results.)
    # 2列目にTrain、3列目にTestのACC値が記載 
    # (The ACC value of Train is listed in the 2nd column and the ACC value of Test is listed in the 3rd column.)
    resultrf_path = '/home/toseidb/test/resultrf.csv'
    if os.path.exists(resultrf_path):
        df_svmresult = pd.read_csv(resultrf_path)
        df_rfdemo = df_svmresult.values
        # train,valの平均 (mean)
        res_mean = np.mean(df_rfdemo[:,2:],axis = 1)
        # 5000番目までの位置を取得 (Get position up to 5000th)
        tarloc = np.where(res_mean > (sorted(res_mean.ravel())[-5000]))


    # 症例番号を削除したctcal_bycaseを作成 (Create ctcal_bycase with case number removed)
    l_ex = [s*(3*5+1)-1 for s in [1,2,3,4]]
    l_temp = [s for s in list(range((3*5+1)*4)) if not s in l_ex]

    ctcal_bycase_f4 = (ctcal_bycase[:,l_temp]).astype('float32')
    ctcal_testbycase_f4 = (ctcal_testbycase[:,l_temp]).astype('float32')


    # l_n_estimators = [2,4,8,16,32,64, 128]
    for i_loc in tqdm(tarloc[0], total = len(tarloc[0])):
        # 分割毎にRFして結果を出力する (RF for each division and output the result)
        arr_temp = np.empty((0,ctcal_bycase_f4.shape[0]), float)        
        arr_testtemp = np.empty((0,ctcal_testbycase_f4.shape[0]), float)
        l_size = [len(l_dis[combvec[i_loc][s]]) for s in range(4)] # ex. [2, 3, 2, 2]
        l_tempname = ['ipf_L','nonipf_L','ipf_nonipf']

        l_colname = []
        for slice_num in range(4): # sliceをfor loopする
            # そのスライスの分割パターン (the split pattern for that slice)
            l_i = l_dis[combvec[i_loc][slice_num]]
            # [array([1]), array([2, 3, 4]), array([5])]

            l_temp = list(itertools.product(
                    [str(s) for s in list(range(1,len(l_i)+1))], l_tempname)
                    )
            l_colname.append(['slice' + str(slice_num) + 'seg' + '_'.join(s) for s in l_temp ])

            # DICOM1枚のデータにスライス (Slice into one piece of DICOM data)
            # numpyのスライスでは[start:stop]、start <= x < stop。stop番目の値は含まれないので注意!! 
            # (For numpy slices [start:stop], start <= x < stop. Note that the stop-th value is not included!!)
            arr_temp = _ctcalcsum_(ctcal_bycase_f4,slice_num,l_i,arr_temp)
            arr_testtemp = _ctcalcsum_(ctcal_testbycase_f4,slice_num,l_i,arr_testtemp)



        # l_temp = mincomplist + l_mode + l_pca + ['perFVC', 'FEV1FVC', 'perDLCO', 'age', 'gender', 'BMI']
        l_temp = ['age','ANA','BMI','CentromereType','gender','GranularType','HomogeneousType','MPOANCA','NuclearEnvelopeType','NucleolarType','perFVC','PeripheralType','RA','SpeckledType','Neu','Eo','Mac','Lym','antiSCL70','perDLCO','antiSSA_Ab','FEV1FVC','antiSm','antiSSB_Ab','antiARS_Ab','centromere','antiRNPAb','antiDsDNA','CD4CD8','PackYear','antiCCP','IgG','TCC','Jo1']
        l_scale, df_fill = _df_l_(arr_temp,'tosei',ctcal_bycase,l_colname,df_tosei,l_temp)
        l_testscale, df_testfill = _df_l_(arr_testtemp,'KCRC',ctcal_testbycase,l_colname,df_test,l_temp)


        # lstからデータを読み込んでtrainとvalに分ける 
        # (Read data from lst and divide into train and val)
        l_train = [re.match('^[0-9]+', s).group() for s
            in _readtxt_(orgfolder+'lst/train.txt')
            if re.match('^[0-9]+_[0-9]+_e0', s)
            ]
        l_val =  [re.match('^[0-9]+', s).group() for s
            in _readtxt_(orgfolder+'lst/val.txt')
            if re.match('^[0-9]+_[0-9]+_e0', s)
            ]
        l_test =  [re.match('^[0-9]+', s.replace('KCRC','')).group() for s
            in _readtxt_(testfolder+'lst/trainval.txt')
            if re.match('^[0-9]+_[0-9]+_e0', s.replace('KCRC',''))
            ]
            

        X_train, y_train = _pd_np_forML_(l_train,df_fill,l_scale,'tosei')
        X_val, y_val = _pd_np_forML_(l_val,df_fill,l_scale,'tosei')
        X_test, y_test = _pd_np_forML_(l_test,df_testfill,l_testscale,'KCRC')


        acc_train, cu_rf_train = _rftrain_(X_train,y_train)
        acc_val = _rftest_(X_val,y_val,cu_rf_train)
        acc_test = _rftest_(X_test,y_test,cu_rf_train)

        l_svmresult.append([i_loc, acc_train , acc_val, acc_test])

    outpath = efolder+'RF'+ datetime.datetime.now().strftime('%Y%m%d') + 'rapids.csv'
    pd.DataFrame(l_svmresult).to_csv(outpath, index=False)
    print('出力は (The output is):{0}'.format(outpath))



    elapsed_time = time.time() - start
    print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")


if __name__ == "__main__":
    main()
