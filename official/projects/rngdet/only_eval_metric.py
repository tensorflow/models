import os
import json
import time
import shutil
import pickle
from scipy.spatial import cKDTree
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import sys
sys.path.append("/home/mjyun/01_ghpark/models")

import tensorflow as tf
import numpy as np
from PIL import Image, ImageDraw
from official.projects.rngdet.tasks import rngdet
from official.core import exp_factory
exp_config = exp_factory.get_exp_config('rngdet_cityscale')
task_obj = rngdet.RNGDetTask(exp_config.task)
model = task_obj.build_model()
#task_obj.initialize(model)

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)

class Vertex():
    def __init__(self,v,id):
        self.x = v[0]
        self.y = v[1]
        self.id = id
        self.neighbors = []

class Edge():
    def __init__(self,src,dst,id):
        self.src = src
        self.dst = dst
        self.id = id

class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = {}
        self.vertex_num = 0
        self.edge_num = 0

    def find_v(self,v_coord):
        if f'{v_coord[0]}_{v_coord[1]}' in self.vertices.keys():
            return self.vertices[f'{v_coord[0]}_{v_coord[1]}']
        return 

    def find_e(self,v1,v2):
        if f'{v1.id}_{v2.id}' in self.edges:
            return True
        return None

    def add(self,edge):
        v1_coord = edge[0]
        v2_coord = edge[1]
        v1 = self.find_v(v1_coord)
        if v1 is None:
            v1 = Vertex(v1_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v1.x}_{v1.y}'] = v1
        
        v2 = self.find_v(v2_coord)
        if v2 is None:
            v2 = Vertex(v2_coord,self.vertex_num)
            self.vertex_num += 1
            self.vertices[f'{v2.x}_{v2.y}'] = v2

        if v1 not in v2.neighbors:
            v2.neighbors.append(v1)
        if v2 not in v1.neighbors:
            v1.neighbors.append(v2)
        e = self.find_e(v1,v2)
        if e is None:
            self.edges[f'{v1.id}_{v2.id}'] = Edge(v1,v2,self.edge_num)
            self.edge_num += 1
            self.edges[f'{v2.id}_{v1.id}'] = Edge(v2,v1,self.edge_num)
            self.edge_num += 1

def calculate_scores(gt_points,pred_points):
    gt_tree = cKDTree(gt_points)
    if len(pred_points):
        pred_tree = cKDTree(pred_points)
    else:
        return 0,0,0
    thr = 10
    dis_gt2pred,_ = pred_tree.query(gt_points, k=1)
    dis_pred2gt,_ = gt_tree.query(pred_points, k=1)
    
    recall = len([x for x in dis_gt2pred if x<thr])/len(dis_gt2pred)
    acc = len([x for x in dis_pred2gt if x<thr])/len(dis_pred2gt)
    r_f = 0
    if acc*recall:
        r_f = 2*recall * acc / (acc+recall)
    return acc, recall, r_f

def pixel_eval_metric(pred_mask,gt_mask):
    def tuple2list(t):
        return [[t[0][x],t[1][x]] for x in range(len(t[0]))]

    gt_points = tuple2list(np.where(gt_mask!=0))
    pred_points = tuple2list(np.where(pred_mask!=0))

    return calculate_scores(gt_points,pred_points)

#open dataset
with open('./data/dataset/data_split.json','r') as jf:
    tile_list = json.load(jf)['test']

create_directory(f'./segmentation/tests/score',delete=False)
create_directory(f'./segmentation/tests/json',delete=False)
create_directory(f'./segmentation/tests/results/apls',delete=False)
create_directory(f'./segmentation/tests/results/topo',delete=False)


for i, tile_name in enumerate(tile_list):
    print(tile_name)

    pred_graph = np.array(Image.open(f'./segmentation/tests/skeleton/{tile_name}.png'))
    gt_graph = np.array(Image.open(f'./data/dataset/segment/{tile_name}.png'))
    print(pixel_eval_metric(pred_graph,gt_graph))
 
 