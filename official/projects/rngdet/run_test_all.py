import os
import json
import time
import shutil
import pickle
from scipy.spatial import cKDTree
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"

import sys

import tensorflow as tf
import numpy as np

from official.projects.rngdet.tasks import rngdet
from official.core import exp_factory

parser = argparse.ArgumentParser() 
parser.add_argument('--ckpt_dir', '-ckpt', nargs='*', help='ckpt_dir', default=[], dest='ckpt_dir')  

exp_config = exp_factory.get_exp_config('rngdet_cityscale')
task_obj = rngdet.RNGDetTask(exp_config.task)
model = task_obj.build_model()

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
    thr = 5
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

ckpt_dir_or_file = parser.parse_args().ckpt_dir[0] 

ckpt = tf.train.Checkpoint(
    backbone=model.backbone,
    backbone_history=model.backbone_history,
    transformer=model.transformer,
    segment_fpn=model._segment_fpn,
    keypoint_fpn=model._keypoint_fpn,
    query_embeddings=model._query_embeddings,
    segment_head=model._segment_head,
    keypoint_head=model._keypoint_head,
    class_embed=model._class_embed,
    bbox_embed=model._bbox_embed,
    input_proj=model.input_proj)
status = ckpt.restore(tf.train.latest_checkpoint(ckpt_dir_or_file))
status.expect_partial().assert_existing_objects_matched()
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("LOAD CHECKPOINT DONE")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

from PIL import Image, ImageDraw
from official.projects.rngdet.eval import agent

pad_size = 128
logit_threshold = 0.75
roi_size = 128

with open('./data/dataset/data_split.json','r') as jf:
    tile_list = json.load(jf)['test']

time_start = time.time()

create_directory(f'./segmentation/tests/graph',delete=True)
create_directory(f'./segmentation/tests/skeleton',delete=True)
create_directory(f'./segmentation/tests/json',delete=True)

for i, tile_name in enumerate(tile_list):
    print('====================================================')
    print(f'{i}/{len(tile_list)}: Start processing {tile_name}')

    #initialize an agent for current image 

    sat_image = np.array(Image.open(os.path.join('./data/dataset/20cities/region_'+str(tile_name)+'_sat.png')))
    print(f'STEP 1: Initialize agent and extract candidate initial vertices...')
    sat_image = tf.cast(sat_image, tf.float32)
    time1 = time.time()
    agent_new = agent.Agent(model, sat_image)

    print(f'STEP 2: Interative graph detection...')
    while 1:
        agent_new.step_counter += 1
        sat_ROI, historical_ROI = agent_new.crop_ROI(agent_new.current_coord)
        sat_ROI = tf.expand_dims(sat_ROI, 0) / 255.0
        sat_ROI = tf.cast(sat_ROI, tf.float32) 
        historical_ROI = tf.expand_dims(historical_ROI, 0) / 255.0
        historical_ROI = tf.expand_dims(historical_ROI, -1)
        historical_ROI = tf.cast(historical_ROI, tf.float32)

        # predict vertices in the next step
        outputs, pred_segment, pred_keypoint = model(sat_ROI, historical_ROI, training=False)

        # agent_new moves
        # alignment vertices
        pred_coords = outputs['box_outputs']
        pred_probs = outputs['cls_outputs']

        alignment_vertices = [[v[0]-agent_new.current_coord[0]+agent_new.crop_size//2, v[1]-agent_new.current_coord[1]+agent_new.crop_size//2] for v in agent_new.historical_vertices]
        pred_coords_ROI = agent_new.step(pred_probs,pred_coords,thr=logit_threshold)

        # stop action 
        if agent_new.finish_current_image:
            print(f'STEP 3: Finsh exploration. Save visualization and graph...')
            #save historical map
            Image.fromarray(
                agent_new.historical_map[roi_size:-roi_size,roi_size:-roi_size].astype(np.uint8)
                ).convert('RGB').save(f'./segmentation/tests/skeleton/{tile_name}.png')

            graph = Graph()
            try:
                with open(f'./segmentation/tests/json/{tile_name}.json','w') as jf:
                    json.dump(agent_new.historical_edges,jf)
            except Exception as e:
                print('Error...')
                print(e)

            for e in agent_new.historical_edges:
                graph.add(e)

            output_graph = {}

            for k, v in graph.vertices.items():
                output_graph[(v.y,v.x)] = [(n.y,n.x) for n in v.neighbors]

            pickle.dump(output_graph,open(f'./segmentation/tests/graph/{tile_name}.p','wb'),protocol=2)
            pred_graph = np.array(Image.open(f'./segmentation/tests/skeleton/{tile_name}.png'))
            gt_graph = np.array(Image.open(f'./data/dataset/segment/{tile_name}.png'))

            #print score
            print(pixel_eval_metric(pred_graph,gt_graph))

            break

    time2 = time.time()
    print(f'{i}/{len(tile_list)}: Finish processing {tile_name}, time usage {round((time2-time1)/3600,3)}h')

time_end = time.time()
print(f'Finish inference, time usage {round((time_end-time_start)/3600,3)}h')    