import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="7"


import tensorflow as tf
import numpy as np
import argparse 

from official.projects.rngdet.tasks import rngdet
from official.core import exp_factory
exp_config = exp_factory.get_exp_config('rngdet_cityscale')
task_obj = rngdet.RNGDetTask(exp_config.task)
model = task_obj.build_model()

from PIL import Image, ImageDraw
from official.projects.rngdet.eval import agent

parser = argparse.ArgumentParser() 
parser.add_argument('--ckpt_dir', '-ckpt', nargs='*', help='ckpt_dir', default='/home/mjyun_2024/01_master/models/official/projects/rngdet/ckpt/01_pr_ready/' , dest='ckpt_dir')  
ckpt_dir_or_file = parser.parse_args().ckpt_dir  

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


pad_size = 128
sat_image = np.array(Image.open(os.path.join('./region_9_sat.png')))

print(f'STEP 1: Initialize agent and extract candidate initial vertices...')
sat_image = tf.cast(sat_image, tf.float32)
agent = agent.Agent(model, sat_image)

logit_threshold = 0.75
roi_size = 128

print(f'STEP 2: Interative graph detection...')
while 1:
    agent.step_counter += 1
    # crop ROI
    sat_ROI, historical_ROI = agent.crop_ROI(agent.current_coord)

    sat_ROI = tf.expand_dims(sat_ROI, 0) / 255.0
    sat_ROI = tf.cast(sat_ROI, tf.float32) 

    historical_ROI = tf.expand_dims(historical_ROI, 0) / 255.0
    historical_ROI = tf.expand_dims(historical_ROI, -1)
    historical_ROI = tf.cast(historical_ROI, tf.float32)

    # predict vertices in the next step
    outputs, pred_segment, pred_keypoint = model(sat_ROI, historical_ROI, training=False)
    # agent moves
    # alignment vertices
    pred_coords = outputs['box_outputs']
    pred_probs = outputs['cls_outputs']

    #(mj) added from torch code (movement vector)
    alignment_vertices = [[v[0]-agent.current_coord[0]+agent.crop_size//2, v[1]-agent.current_coord[1]+agent.crop_size//2] for v in agent.historical_vertices]
    pred_coords_ROI = agent.step(pred_probs,pred_coords,thr=logit_threshold)

    if agent.step_counter%100==0:
        if agent.step_counter%500==0:
            print(f'Iteration {agent.step_counter}...')
            Image.fromarray(agent.historical_map[roi_size:-roi_size,roi_size:-roi_size].astype(np.uint8)).convert('RGB').save(f'./segmentation/skeleton_result_{agent.step_counter}.png')
            # visualize current step's result
        pred_binary = tf.math.sigmoid(pred_segment[0, :, :, 0]) * 255 #output['pred_masks'] -> pred_segment
        pred_keypoints = tf.math.sigmoid(pred_keypoint[0, :, :, 0]) * 255 #output['pred_masks'] -> pred_keypoint
        # vis
        dst = Image.new('RGB',(roi_size*2+5,roi_size*2+5))

        sat_ROI_tmp = sat_ROI[0]*255 
        history_tmp = historical_ROI[0, :, :, 0]*255

        sat = Image.fromarray(sat_ROI_tmp.numpy().astype(np.uint8))  
        history = Image.fromarray(history_tmp.numpy().astype(np.uint8))  
        pred_binary = Image.fromarray(pred_binary.numpy().astype(np.uint8))  
        pred_keypoint = Image.fromarray(pred_keypoints.numpy().astype(np.uint8))  

        dst.paste(sat,(0,0))  
        dst.paste(history,(0,roi_size))
        dst.paste(pred_binary,(roi_size,0))
        dst.paste(pred_keypoint,(roi_size,roi_size)) 
        draw = ImageDraw.Draw(dst)

        for ii in range(3):  
            for kk in range(2):  
                delta_x = ii*roi_size
                delta_y = kk*roi_size
                if len(alignment_vertices):  
                    for v in alignment_vertices:
                        if v[0]>=0 and v[0]<agent.crop_size and v[1]>=0 and v[1]<agent.crop_size:
                            v = [delta_x+(v[0]),delta_y+(v[1])]
                            draw.ellipse((v[0]-1,v[1]-1,v[0]+1,v[1]+1),fill='cyan',outline='cyan')

                if pred_coords_ROI: 
                    for jj in range(len(pred_coords_ROI)):
                        v = pred_coords_ROI[jj]
                        v = [delta_x+(v[0]),delta_y+(v[1])]
                        draw.ellipse((v[0]-1,v[1]-1,v[0]+1,v[1]+1),fill='pink',outline='pink')

                draw.ellipse([delta_x-1+roi_size//2,delta_y-1+roi_size//2,delta_x+1+roi_size//2,delta_y+1+roi_size//2],fill='orange')
        dst.convert('RGB').save(f'./segmentation/vis_result_{agent.step_counter}.png') 
        #exit()

    if agent.finish_current_image:
        print(f'STEP 3: Finsh exploration. Save visualization and graph...')
        #save historical map
        Image.fromarray(
            agent.historical_map[roi_size:-roi_size,roi_size:-roi_size].astype(np.uint8)
            ).convert('RGB').save(f'./segmentation/9_result.png')

        break