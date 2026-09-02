from ctypes import alignment
import enum
import numpy as np
import os
import random
import json
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
from skimage import measure
import tensorflow as tf

class FrozenClass():
        __isfrozen = False
        def __setattr__(self, key, value):
            if self.__isfrozen and not hasattr(self, key):
                raise TypeError( "%r is a frozen class" % self )
            object.__setattr__(self, key, value)

        def _freeze(self):
            self.__isfrozen = True

class Agent(FrozenClass):
    def __init__(self, model, image):
        # args
        self.image_size = 2048
        self.crop_size = 128
        self.pad_size = 128
        self.extract_candidate_threshold = 0.65
        self.candidate_filter_threshold = 25
        self.alignment_distance = 10
        self.process_boundary = False
        # counters 
        self.step_counter = 0
        self.LS_counter = 0
        # data
        self.sat_image = None
        self.vertices = {}
        self.edges = {}
        self.model = model
        self.image = image
        # state
        self.mode = 'vertex_mode'
        self.finish_current_image = False
        self.current_coord = None
        self.previous_coord = None
        self.counter_map = np.zeros((self.image_size,self.image_size))
        # historical info
        self.historical_map = np.zeros((self.image_size,self.image_size), np.float32)
        self.historical_map = np.pad(self.historical_map,np.array(((self.pad_size,self.pad_size),(self.pad_size,self.pad_size))),'constant')
        self.historical_vertices = []
        self.historical_edges = []
        # buffer
        self.candidate_initial_vertices = []
        self.extracted_candidate_initial_vertices = []

        self.load_data_and_initialization()
        self.__setup_seed(20)
        self._freeze()

    def __setup_seed(self,seed):
        '''
        Random seed
        :param seed (int): seed
        '''
        np.random.seed(seed)
        random.seed(seed)

    def load_data_and_initialization(self):
        '''
        Load data and initialize env.
        :param image_name (str): name of input image
        '''
        self.sat_image = np.pad(self.image,np.array(((self.pad_size,self.pad_size),(self.pad_size,self.pad_size),(0,0))),'constant')

        # get initial candidates
        self.get_candidate_initial_vertices_from_local_peaks()
        # initialization
        self.pop()

    def get_candidate_initial_vertices_from_local_peaks(self):
        '''
        Generate candidate initial vertices from predicted point segmentation heatmap 
        :param image_name (str): name of input image
        '''
        sigmoid_fn = tf.keras.layers.Activation("sigmoid")
        binary_mask = np.zeros((self.image_size+self.pad_size*2,self.image_size+self.pad_size*2))
        point_mask = np.zeros((self.image_size+self.pad_size*2,self.image_size+self.pad_size*2))
        for i in range(self.image_size//self.crop_size+1):
            for j in range(self.image_size//self.crop_size+1):
                sat_ROI = self.sat_image[self.crop_size//2+i*self.crop_size:self.crop_size//2+(i+1)*self.crop_size,\
                        self.crop_size//2+j*self.crop_size:self.crop_size//2+(j+1)*self.crop_size]
                historical_ROI = self.historical_map[self.crop_size//2+i*self.crop_size:self.crop_size//2+(i+1)*self.crop_size,\
                        self.crop_size//2+j*self.crop_size:self.crop_size//2+(j+1)*self.crop_size]
                sat_ROI = tf.expand_dims(sat_ROI, 0) / 255.0 

                historical_ROI = tf.expand_dims(historical_ROI, 0)  
                historical_ROI = tf.expand_dims(historical_ROI, -1)
                historical_ROI = tf.cast(historical_ROI, tf.float32)

                _, pred_segment, pred_keypoint = self.model(sat_ROI, historical_ROI, training=False)
                pred_mask = tf.cast(  ( sigmoid_fn(pred_segment) ) *255, tf.uint8)
                pred_point = tf.cast( ( sigmoid_fn(pred_keypoint) )*255, tf.uint8)

                binary_mask[self.crop_size//2+i*self.crop_size:self.crop_size//2+(i+1)*self.crop_size,\
                        self.crop_size//2+j*self.crop_size:self.crop_size//2+(j+1)*self.crop_size] = tf.squeeze(pred_mask).numpy()
                point_mask[self.crop_size//2+i*self.crop_size:self.crop_size//2+(i+1)*self.crop_size,\
                        self.crop_size//2+j*self.crop_size:self.crop_size//2+(j+1)*self.crop_size] = tf.squeeze(pred_point).numpy()
        self.extract_initial_candidates(point_mask[self.pad_size:-self.pad_size,self.pad_size:-self.pad_size],thr=self.extract_candidate_threshold)

        image = Image.fromarray(point_mask[self.pad_size:-self.pad_size,self.pad_size:-self.pad_size].astype(np.uint8)).convert('RGB')
        draw = ImageDraw.Draw(image)
        for v in self.extracted_candidate_initial_vertices:
            draw.ellipse([v[0]-5,v[1]-5,v[0]+5,v[1]+5],fill='red')
        image.convert('RGB').save(f'./segmentation/initialized.png') 
        with open(f'./segmentation/init_result.json','w') as jf:
            json.dump(self.extracted_candidate_initial_vertices,jf)

        print(f'{len(self.extracted_candidate_initial_vertices)} initial candidates extracted from the segmentation map...')
        if len(self.extracted_candidate_initial_vertices)==0:
            self.finish_current_image = True

    def extract_initial_candidates(self,image,thr=0.5):
        '''
        Extract initial vertex candidates from point segmentation heatmap
        :param image (np.array, self.image_size*self.image_size): a world coord
        :param thr (float): threshold for thresholding
        '''
        image = (image>thr*255).astype(np.uint8)
        labels = measure.label(image, connectivity=2)
        props = measure.regionprops(labels)
        max_area = 16
        for region in props:
            if region.area > max_area:
                center_point = region.centroid[::-1]
                self.extracted_candidate_initial_vertices.append([int(x) for x in center_point])

    def crop_ROI(self,v):
        '''
        Crop ROI
        :param v (list, length=2): a world coord
        :return sat_ROI (np.array, size=(self.crop_size,self.crop_size,3)): cropped sat image
        :return historical_map_ROI (np.array, size=(self.crop_size,self.crop_size,3)): cropped historical map
        '''
        sat_ROI = self.sat_image[ v[1]+self.crop_size//2:v[1]+self.crop_size//2*3, v[0]+self.crop_size//2:v[0]+self.crop_size//2*3,: ]
        historical_map_ROI = self.historical_map[ v[1]+self.crop_size//2:v[1]+self.crop_size//2*3 , v[0]+self.crop_size//2:v[0]+self.crop_size//2*3 ]

        return sat_ROI, historical_map_ROI

    def filter_candidate_initial_vertices(self):
        '''
        Remove candidate initial vertices that too closed to historical graph (explored pixels)
        '''
        explored_pixels = np.where(self.historical_map[self.pad_size:-self.pad_size,self.pad_size:-self.pad_size]!=0)
        explored_pixels = [[explored_pixels[1][x], explored_pixels[0][x]] for x in range(len(explored_pixels[0]))]
        if len(explored_pixels):
            tree = cKDTree(explored_pixels)
            if len(self.extracted_candidate_initial_vertices):
                dds, iis = tree.query(self.extracted_candidate_initial_vertices,k=1)
                self.extracted_candidate_initial_vertices = [v for i,v in enumerate(self.extracted_candidate_initial_vertices) if dds[i]>self.candidate_filter_threshold]

    def pop(self):
        '''
        Pop one candidate initial vertex from the buffer
        '''
        if not len(self.candidate_initial_vertices):
            self.filter_candidate_initial_vertices()
            if len(self.extracted_candidate_initial_vertices):
                self.candidate_initial_vertices.append(self.extracted_candidate_initial_vertices.pop(0))
            else:
                self.finish_current_image = True
                return

        self.mode = 'vertex_mode'
        self.current_coord = self.candidate_initial_vertices.pop(0)
        return


    def update_historical_map(self,src,dst):
        '''
        Update the historical map by adding a line starting from src to dst.
        :param src (list, length=2): src point of the added line
        :param dst (list, length=2): src point of the added line
        '''
        src = np.array(src)
        dst = np.array(dst)
        self.counter_map[dst[1],dst[0]] += 1 
        p = src
        d = dst - src
        N = np.max(np.abs(d))
        self.historical_map[src[1]+self.pad_size,src[0]+self.pad_size] = 255

        self.historical_map[dst[1]+self.pad_size,dst[0]+self.pad_size] = 255

        if N:
            s = d / (N)
            for i in range(0,N):
                p = p + s
                self.historical_map[int(round(p[1]+self.pad_size)),int(round(p[0]+self.pad_size))] = 255


    def get_valid_coords(self,pred_logits,pred_coords,thr=0.8):
        '''
        Convert predictions to vertex coordinates.
        :param pred_logits (tensor, shape=(1,args.num_queries,2)): logits of predicted vertices
        :param pred_coords (tensor, shape=(1,args.num_queries,2)): coords of predicted vertices
        :param thr (float): threshold for logits thresholding
        :return output_pred_coords_ROI(list): valid predicted vertices in the next step in the ROI coordinate (cropped ROI whose size is args.crop_size*self.crop_size)
        :return output_pred_coords_world(list): valid predicted vertices in the next step in the world coordinate 
        '''

        softmax_fn = tf.keras.layers.Activation("softmax")

        # transformation within ROI
        pred_coords_ROI = tf.stop_gradient(pred_coords[0]).numpy().tolist()
        pred_coords_ROI = [[x[0]*(self.crop_size//2)+self.crop_size//2,x[1]*(self.crop_size//2)+self.crop_size//2] for x in pred_coords_ROI]

        # find valid coords
        pred_logits = softmax_fn(pred_logits[0])

        temp_pred_coords_ROI = []
        for ii, coord in enumerate(pred_coords_ROI):
            if self.process_boundary and (coord[0]+self.current_coord[0]-self.crop_size//2<=75 \
                or coord[0]+self.current_coord[0]-self.crop_size//2>=self.image_size-75 \
                or coord[1]+self.current_coord[1]-self.crop_size//2<=75 \
                or coord[1]+self.current_coord[1]-self.crop_size//2>=self.image_size-75):
                if pred_logits[ii][0] >= thr + 0.1:
                    temp_pred_coords_ROI.append(coord)
            # 100% this 
            else:
                if pred_logits[ii][0] >= thr:
                    temp_pred_coords_ROI.append(coord)

        temp_pred_coords_ROI = [[max(0,min(self.crop_size-1,int(y))) for y in x] for x in temp_pred_coords_ROI if x[0]>=0 and x[0]<=self.crop_size-1 and x[1]>=0 and x[1]<=self.crop_size-1]

        # filter coord by angle
        pred_coords_ROI = temp_pred_coords_ROI.copy()
        for ii, v in enumerate(temp_pred_coords_ROI[:-1]):
            for u in temp_pred_coords_ROI[ii+1:] :
                vector_v = np.array(v) - np.array([self.crop_size//2,self.crop_size//2])
                vector_u = np.array(u) - np.array([self.crop_size//2,self.crop_size//2])
                norm_v = np.linalg.norm(vector_v)
                norm_u = np.linalg.norm(vector_u)
                if not norm_v:
                    if v in pred_coords_ROI:
                        pred_coords_ROI.remove(v)
                else:
                    vector_v = vector_v / norm_v
                if not norm_u:
                    if u in pred_coords_ROI:
                        pred_coords_ROI.remove(u)
                else:
                    vector_u = vector_u / norm_u
                #if ( ( vector_v.dot(vector_u) > 0.999 ) or ( vector_v.dot(vector_u) < 0 ) )  and norm_u and norm_v:
                if ( ( vector_v.dot(vector_u) > 0.999 ))  and norm_u and norm_v:
                    if u in pred_coords_ROI:
                        pred_coords_ROI.remove(u)

        # from ROI to world
        pred_coords_world = [[int(x[0]+self.current_coord[0]-self.crop_size//2),int(x[1]+self.current_coord[1]-self.crop_size//2)] for x in pred_coords_ROI]
        pred_coords_world = [v for v in pred_coords_world if v[0]>=35 and v[1]>=35 and v[0]<self.image_size-35 and v[1]<self.image_size-35]
        pred_coords_world = [[min(self.image_size,max(0,y)) for y in x] for x in pred_coords_world]

        # alignment
        if len(self.historical_vertices):
            tree = cKDTree(self.historical_vertices)
            for i, v in enumerate(pred_coords_world):
                dd, ii = tree.query(v,k=1) #distance between pred_coords_world's neareast historical point 
                if ( dd < self.alignment_distance ) and ( self.historical_vertices[ii]!=self.current_coord ):
                    pred_coords_world[i] = self.historical_vertices[ii] #choose one of the historical vertices

        # remove coords that are not moving or filtered
        output_pred_coords_world = [v for i,v in enumerate(pred_coords_world) if v!=self.current_coord and self.counter_map[v[1],v[0]]<10]

        # world to ROI
        output_pred_coords_ROI = [[v[0]-self.current_coord[0]+self.crop_size//2,v[1]-self.current_coord[1]+self.crop_size//2] for v in output_pred_coords_world]
        return output_pred_coords_ROI, output_pred_coords_world


    def step(self, pred_logits, pred_coords, thr=0.9):
        '''
        Agent moves function. There could be zero, one or multiple predicted vertices in the next step. The agent should take
        corresponding actions depending on the predictions.
        :param pred_logits (tensor, shape=(1,args.num_queries,2)): logits of predicted vertices
        :param pred_coords (tensor, shape=(1,args.num_queries,2)): coords of predicted vertices
        :param thr (float): threshold for logits thresholding
        :return pred_coords_ROI (list): valid predicted vertices in the next step in the ROI coordinate
        '''
        pred_coords_ROI, pred_coords_world = self.get_valid_coords(pred_logits, pred_coords, thr=thr)
        # intersection mode
        self.LS_counter += 1
        # zero prediction
        if not len(pred_coords_world) or self.LS_counter>30:
            self.LS_counter = 0
            if self.current_coord[0]>=50 and self.current_coord[0]<=self.image_size-50 and self.current_coord[1]>=50 and self.current_coord[1]<=self.image_size-50:
                self.historical_vertices.append(self.current_coord)
            self.pop()
            return
        # multiple predictions
        elif len(pred_coords_world) > 1:
            # transformation
            for ii, pred_coord_world in enumerate(pred_coords_world):
                self.candidate_initial_vertices.append(pred_coord_world)
                self.update_historical_map(self.current_coord,pred_coord_world) #(self, src, dst)
                self.historical_edges.append([[int(x) for x in self.current_coord],[int(x) for x in pred_coord_world]])
                self.historical_vertices.append(pred_coord_world)
            self.historical_vertices.append(self.current_coord)
            self.pop()
            return pred_coords_ROI
        # line-segment mode, one prediction
        elif len(pred_coords_world) == 1:
            pred_coord_world = pred_coords_world[0]
            self.previous_coord = self.current_coord
            self.current_coord = pred_coord_world
            self.update_historical_map(self.current_coord,self.previous_coord)
            self.historical_edges.append([[int(x) for x in self.current_coord],[int(x) for x in self.previous_coord]])
            return pred_coords_ROI
