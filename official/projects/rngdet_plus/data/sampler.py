import time
import numpy as np
#import torch
import os
import random
import json
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw
import random

class FrozenClass():
        __isfrozen = False
        def __setattr__(self, key, value):
            if self.__isfrozen and not hasattr(self, key):
                raise TypeError( "%r is a frozen class" % self )
            object.__setattr__(self, key, value)

        def _freeze(self):
            self.__isfrozen = True

class Edge(FrozenClass):
  def __init__(self,edge_move_ahead_length,id,src,dst,vertices,orientation):
    self.edge_move_ahead_length = edge_move_ahead_length
    self.id = id
    # init and end
    self.src = src
    self.dst = dst
    # state
    self.tracked = False
    self.finished = False
    self.v_now = vertices[0].copy()
    self.max_index = 0
    # data
    self.vertices = vertices
    self.raw_vertices = vertices.copy()
    self.orientation = orientation
    self.raw_orientation = orientation.copy()
    self.ahead_segment = []

  def reverse(self):
    self.src, self.dst = self.dst, self.src
    self.vertices = self.vertices[::-1]
    # self.tree = cKDTree(self.vertices)
    self.orientation = self.orientation[::-1]
    self.v_now = self.vertices[0].copy()
    self.max_index = 0
    self.ahead_segment = []

  def step_BC(self,init_step=False):
    edge_move_ahead_length = self.edge_move_ahead_length if not init_step else self.edge_move_ahead_length // 2
    # find index now
    tree = cKDTree(self.vertices[self.max_index:self.max_index+2*edge_move_ahead_length])
    dds, iis = tree.query(self.v_now,k=1)
    index_now = iis + self.max_index
    if index_now + edge_move_ahead_length >= len(self.vertices):
      # end vertex too closed
      v_next = self.vertices[-1]
      index_next = len(self.vertices) - 1
      self.finished = True
    else:
      # Agent moves along the current edge
      orientation = self.orientation[index_now:index_now+edge_move_ahead_length]
      curve_vertex_index = [i for i,x in enumerate(orientation) if abs(x-orientation[0])>2]
      if len(curve_vertex_index):
        # Curve ahead
        index_next = index_now + curve_vertex_index[0] + 1
      else:
        # Straight ahead
        index_next = index_now + edge_move_ahead_length
      index_next = max(index_next,self.max_index)
      v_next = self.vertices[index_next]
    self.v_now = v_next
    self.max_index = index_next
    self.ahead_segment = self.vertices[index_now:index_next]
    # update edge
    if init_step:
      self.src = Vertex(-1,v_next[0],v_next[1],neighbors=[self])
      self.vertices = self.vertices[index_next:]
      self.orientation = self.orientation[index_next:]
      self.max_index = 0
      self.v_now = self.vertices[0]
    return v_next




class Vertex(FrozenClass):
  def __init__(self,id,x,y,neighbors=None):
    self.id = id
    self.x = x
    self.y = y
    self.neighbors = [] if not neighbors else neighbors
    self.visited = False
    self._freeze()



class Sampler(FrozenClass):
  def __init__(self, dataroot, image_size, roi_size, num_queries,
               noise, edge_move_ahead_length, image_name):
    # args
    self.image_size = image_size
    self.crop_size = roi_size
    self.pad_size = roi_size
    self.dataroot = dataroot
    self.num_queries = num_queries
    self.noise = noise
    self.edge_move_ahead_length = edge_move_ahead_length
    # counters 
    self.step_counter = 0
    # data
    self.sat_image = None
    self.label_masks = np.zeros((self.image_size,self.image_size,2))
    self.vertices = {}
    self.edges = {}
    # state
    self.mode = 'vertex_mode'
    self.current_v = None
    self.current_e = None
    self.finish_current_image = False
    self.current_coord = None
    # historical info
    self.historical_map = np.zeros((self.image_size,self.image_size))
    self.historical_map = np.pad(self.historical_map,np.array(((self.pad_size,self.pad_size),(self.pad_size,self.pad_size))),'constant')
    self.historical_vertices = []
    self.historical_edges = []
    # buffer
    self.candidate_initial_vertices = []

    self.load_data_and_initialization(image_name)
    self.__setup_seed(20)
    self._freeze()

  def __setup_seed(self,seed):
    '''
    Random seed
    :param seed (int): seed
    '''
    #torch.manual_seed(seed)
    #torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True

  def reset_env(self):
    # counters 
    self.step_counter = 0
    # data
    self.sat_image = None
    self.label_masks = np.zeros((self.image_size,self.image_size,2))
    self.vertices = {}
    self.edges = {}
    # state
    self.mode = 'vertex_mode'
    self.current_v = None
    self.current_e = None
    self.finish_current_image = False
    # historical info
    self.historical_map = np.zeros((self.image_size,self.image_size))
    self.historical_map = np.pad(self.historical_map,np.array(((self.pad_size,self.pad_size),(self.pad_size,self.pad_size))),'constant')
    self.historical_vertices = []
    self.historical_edges = []
    # buffer
    self.candidate_initial_vertices = [] 

  def load_data_and_initialization(self,tile_name):
    '''
    Load data and initialize sampler.
    :param image_name (str): name of input image
    '''
    self.reset_env()
    # load images
    self.sat_image = np.array(Image.open(os.path.join(self.dataroot,f'20cities/region_{tile_name}_sat.png')))
    self.sat_image = np.pad(self.sat_image,np.array(((self.pad_size,self.pad_size),(self.pad_size,self.pad_size),(0,0))),'constant')
    # segment_mask
    self.label_masks[:,:,0] = np.array(Image.open(os.path.join(self.dataroot,f'segment/{tile_name}.png')))[:,:,0]
    # point_mask
    self.label_masks[:,:,1] = np.array(Image.open(os.path.join(self.dataroot,f'intersection/{tile_name}.png')))[:,:,0]
    self.label_masks = np.pad(self.label_masks,np.array(((self.pad_size,self.pad_size),(self.pad_size,self.pad_size),(0,0))),'constant')

    # load graph
    with open(os.path.join(self.dataroot,f'graph/{tile_name}.json'),'r') as jf:
      json_data = json.load(jf)
    vertices = json_data['vertices']
    for i, v in enumerate(vertices):
      new_v = Vertex(v['id'],v['x'],v['y'])
      self.vertices[v['id']] = new_v
    edges = json_data['edges']
    for i, e in enumerate(edges):
      src, dst = self.vertices[e['src']], self.vertices[e['dst']]
      new_edge = Edge(self.edge_move_ahead_length,e['id'],src,dst,e['vertices'],e['orientation'])
      self.edges[e['id']] = new_edge
      src.neighbors.append(new_edge)
      dst.neighbors.append(new_edge)
      pass
    # get initial candidates
    for k,v in self.vertices.items():
      if len(v.neighbors)==1:
        self.candidate_initial_vertices.append(v)
    # initialization
    self.pop()


  def update_historical_map(self,src,dst):
    '''
    Update the historical map by adding a line starting from src to dst.
    :param src (list, length=2): src point of the added line
    :param dst (list, length=2): src point of the added line
    '''
    src = np.array(src)
    dst = np.array(dst)
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

  def crop_ROI(self,v):
    '''
    Crop ROI
    :param v (list, length=2): a world coord
    :return sat_ROI (np.array, size=(self.crop_size,self.crop_size,3)): cropped sat image
    :return historical_map_ROI (np.array, size=(self.crop_size,self.crop_size,3)): cropped historical map
    '''
    sat_ROI = self.sat_image[v[1]+self.crop_size//2:v[1]+self.crop_size//2*3,v[0]+self.crop_size//2:v[0]+self.crop_size//2*3,:]
    label_masks_ROI = self.label_masks[v[1]+self.crop_size//2:v[1]+self.crop_size//2*3,v[0]+self.crop_size//2:v[0]+self.crop_size//2*3,:]
    historical_map_ROI = self.historical_map[v[1]+self.crop_size//2:v[1]+self.crop_size//2*3,v[0]+self.crop_size//2:v[0]+self.crop_size//2*3].copy()

    return sat_ROI, label_masks_ROI, historical_map_ROI

  def pop(self):
    '''
    Pop one candidate initial vertex from the buffer
    '''
    while 1:
      if len(self.candidate_initial_vertices):
        self.mode = 'edge_mode'
        candidate_initial_vertex = self.candidate_initial_vertices.pop(0)
        self.current_v = candidate_initial_vertex
        self.current_coord = [self.current_v.x,self.current_v.y]
        if len(self.current_v.neighbors)!=1:
            raise Exception('Incorrect number of incident edges!')
        self.current_e = self.current_v.neighbors[0]
        if self.current_e.finished:
            continue
        if self.current_e.src!=self.current_v:
            self.current_e.reverse()
            # continue
        self.current_e.tracked = True
        return 
      else:
        self.finish_current_image = True
        return

  def calcualte_label(self,v_current,v_nexts):
    '''
    Based on the v_current and v_nexts, calculate coordinate and prob training labels. 
    :param v_current (list, size=(1,2)): current vertex coord
    :param v_nexts (list, size=(n,2)): gt vertex coords in the next step (generated by expert)
    :return output_prob (np.array, size=(args.num_queries)): probability label
    :return output_coord (np.array, size=(args.num_queries,2)): coordinates label
    :return list_len (int): number of valid vertices in the next step
    '''
    output_prob = np.ones((self.num_queries))
    output_coord = np.ones((self.num_queries,2))
    list_len = 0
    gt_coords = []
    gt_probs = []
    v_current = np.array(v_current)
    for v_next in v_nexts:
      vector = v_next - v_current
      vector = [(x)/(self.crop_size//2) for x in vector]
      gt_coords.append(vector)
      gt_probs.append(0)
    gt_coords,gt_probs = np.array(gt_coords),np.array(gt_probs)
    list_len = gt_probs.shape[0]
    if list_len:
      output_prob[:list_len] = gt_probs
      output_coord[:list_len] = gt_coords
    return output_prob, output_coord, list_len

  def step_expert_BC_sampler(self):
    '''
    Control the agent explore the graph to generate training samples (behaviro cloning). Calculate vertices in the next step.
    :return v_nexts (list, size=n*2): vertices in the next step
    '''
    v_nexts = []
    ahead_segments = []
    self.step_counter += 1
    # vertex mode, find vertices in the next step
    if self.mode == 'vertex_mode':
      unexplored_edges = [e for e in self.current_v.neighbors if not e.finished]
      if len(unexplored_edges):
        # find the label of each unexplore edge
        for e in unexplored_edges:
          if e.src!=self.current_v:
            e.reverse()
            # continue
          v_next = e.step_BC(init_step=True)
          self.update_historical_map(self.current_coord,v_next)
          if not e.tracked:
            self.candidate_initial_vertices.append(e.src)
          e.tracked = True
          v_nexts.append(v_next)
          ahead_segments.append([[v[0]-self.current_coord[0]+self.crop_size//2,v[1]-self.current_coord[1]+self.crop_size//2] for v in e.ahead_segment])
      self.pop()

    # edge mode, find vertex in the next step
    elif self.mode == 'edge_mode':
      v_next = self.current_e.step_BC()
      v_nexts = [v_next]
      ahead_segments = [[[v[0]-self.current_coord[0]+self.crop_size//2,v[1]-self.current_coord[1]+self.crop_size//2] for v in self.current_e.ahead_segment]]
      if self.current_e.finished:
        self.mode = 'vertex_mode'
        self.current_v = self.current_e.dst
        self.update_historical_map(self.current_coord,v_next)
        self.current_coord = v_next
      else:
        # add unique noise
        v_next_noise = np.array(v_next) + np.random.uniform(-1, 1, 2)*self.noise
        v_next_noise = [int(max(0,min(self.image_size-1,x))) for x in v_next_noise]
        self.current_e.v_now = v_next_noise
        self.update_historical_map(self.current_coord,v_next_noise)
        self.current_coord = v_next_noise

    else:
      raise Exception('Incorrect mode name!!!')
    for segment in ahead_segments:
      for v in segment:
        if v[0]<0 or v[1]<0 or v[0]>=128 or v[1]>=128:
          print(ahead_segments)
          raise Exception
    return v_nexts, ahead_segments
