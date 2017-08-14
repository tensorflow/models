# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cPickle as pickle
import os

def save(stuff, path, info):
    with open(path, 'wb') as output:
        pickle.dump(stuff, output, pickle.HIGHEST_PROTOCOL)
    print (info)


def load(path, info=None):
    with open(path, 'rb') as input:
        stuff = pickle.load(input)
    print (info)
    return stuff


def writeflag(path, flag, info=None):
    with open(path, 'wb') as the_file:
        the_file.write(str(flag))
    print(info)


def readflag(path):
    with open(path, 'rb') as the_file:
        all = the_file.read()
        print ('================================  Flag')
        print (all)
        flag = all[0][0]
        flag = int(flag)
    return flag


def createfolder(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    print ('%s is created' %directory)

def cleartfrecord(path, pattern='/train-*****-of-00008', info= None):
    try:
        os.system('rm %s%s'%path,pattern)
        print ('%s%s are removed :D'%path,pattern)
        print (info)
    except:
        pass


