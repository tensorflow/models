import json 
import os
from PIL import Image, ImageDraw
import numpy as np 
import json 
from tqdm import tqdm
import shutil
import pickle

IMAGE_SIZE = 2048
INTER_P_RADIUS = 1
SEGMENT_WIDTH = 3

def create_directory(dir,delete=False):
    if os.path.isdir(dir) and delete:
        shutil.rmtree(dir)
    os.makedirs(dir,exist_ok=True)

create_directory('./dataset/vis',delete=True)
create_directory('./dataset/graph',delete=True)
create_directory('./dataset/intersection',delete=True)
create_directory('./dataset/segment',delete=True)

class Graph():
    def __init__(self):
        self.vertices = {}
        self.edges = []
        self.edge_counter = 0
        self.v_id = 0
        self.nidmap = None

    def add_v(self,v_list):
        if f'{v_list[0]}_{v_list[1]}' in self.vertices.keys():
            return self.vertices[f'{v_list[0]}_{v_list[1]}']
        else:
            self.vertices[f'{v_list[0]}_{v_list[1]}'] = Vertex(v_list[0],v_list[1],self.v_id)
            self.v_id += 1
            return self.vertices[f'{v_list[0]}_{v_list[1]}']

    def add_e(self,v_list1,v_list2):
        v1 = self.add_v(v_list1)
        v2 = self.add_v(v_list2)
        if v1 not in v2.neighbor_vertices:
            v1.neighbor_vertices.append(v2)
            v2.neighbor_vertices.append(v1)
            new_edge = Edge(v1,v2,self.edge_counter)
            self.edge_counter += 1
            v1.neighbor_edges.append(new_edge)
            v2.neighbor_edges.append(new_edge)
            self.edges.append(new_edge)


    def merge(self,edge_list):
        e1 = edge_list[0]
        e2 = edge_list[1]
        # circle
        if (e1.dst == e2.src and e1.src == e2.dst) or (e1.dst == e2.dst and e1.src == e2.src):
            return
        # no circle
        if e1.dst == e2.src:
            pass
        elif e1.src == e2.src:
            e1.reverse()
        elif e1.dst == e2.dst:
            e2.reverse()
        elif e1.src == e2.dst:
            e1, e2 = e2, e1
        else:
            raise Exception('Error edge vertices...')

        new_e = Edge(e1.src,e2.dst,self.edge_counter,vertices=(e1.vertices[:-1]+e2.vertices))
        self.edge_counter += 1
        e1.src.neighbor_edges.remove(e1)
        e1.src.neighbor_edges.append(new_e)
        e1.dst.removed = True
        e2.dst.neighbor_edges.remove(e2)
        e2.dst.neighbor_edges.append(new_e)
        self.edges.remove(e1)
        self.edges.remove(e2)
        self.edges.append(new_e)

class Vertex():
    def __init__(self,x,y,id):
        self.x = x
        self.y = y
        self.neighbor_vertices = []
        self.neighbor_edges = []
        self.id = id
        self.removed = False


class Edge():
    def __init__(self,src,dst,id,vertices=None,orientation=None):
        self.id = id
        self.src = src
        self.dst = dst
        if not vertices:
            self.vertices = [[src.x,src.y],[dst.x,dst.y]]
        else:
            self.vertices = vertices
        self.orientation = orientation

    def reverse(self):
        self.src, self.dst = self.dst, self.src
        self.vertices = self.vertices[::-1]


def get_orientation_angle(vector):
    norm = np.linalg.norm(vector)
    theta = 0
    if norm:
        vector = vector / norm
        theta = np.arccos(vector[0])
        if vector[1] > 0:
            theta = 2*np.pi - theta
        theta = (theta//(np.pi/32))%64 + 1
    return int(theta)     


def get_dense_edge(src,dst):
    start_vertex = np.array([int(src[0]),int(src[1])])
    end_vertex = np.array([int(dst[0]),int(dst[1])])
    p = start_vertex
    d = end_vertex - start_vertex
    N = np.max(np.abs(d))
    output_list = [[int(x) for x in start_vertex.tolist()]]
    if N:
        s = d / (N)
        for i in range(0,N):
            p = p + s
            p_list = [int(x+0.5) for x in p]
            if p_list not in output_list:
                output_list.append(p_list)
    return output_list

def whether_inside(v):
    if not isinstance(v, list):
        v = [v.x,v.y]
    if v[0]<IMAGE_SIZE and v[0]>=0 and v[1]<IMAGE_SIZE and v[1]>=0:
        return True
    return False

def interpolation(v_out,v_in):
    if (v_out[0])*(v_in[0]) < 0: 
        v_out[0] = 0
        v_out[1] = v_out[1]-abs(v_out[0])*abs(v_in[1]-v_out[1])/abs(v_out[0]-v_in[0])
    elif (v_out[0]-IMAGE_SIZE)*(v_in[0]-IMAGE_SIZE) < 0:
        v_out[0] = IMAGE_SIZE-1
        v_out[1] = v_out[1]-abs(v_out[0]-IMAGE_SIZE)*abs(v_in[1]-v_out[1])/abs(v_out[0]-v_in[0])
    elif (v_out[1])*(v_in[1]) < 0:
        v_out[1] = 0
        v_out[0] = v_out[0]-abs(v_out[1])*abs(v_in[0]-v_out[0])/abs(v_out[1]-v_in[1])
    elif (v_out[1]-IMAGE_SIZE)*(v_in[1]-IMAGE_SIZE) < 0:
        v_out[1] = IMAGE_SIZE-1
        v_out[0] = v_out[0]-abs(v_out[1]-IMAGE_SIZE)*abs(v_in[0]-v_out[0])/abs(v_out[1]-v_in[1])
    return [int(x) for x in v_out]

for tile_index in range(180):
    vertices = []
    edges = []
    vertex_flag = True

    # =============================================================================================
    # Data load part
    # =============================================================================================

    gt_graph = pickle.load(open(f"./dataset/20cities/region_{tile_index}_refine_gt_graph.p",'rb'))
    graph = Graph()
    for n, v in gt_graph.items():
        for nei in v:
            graph.add_e([int(n[1]), int(n[0])],[int(nei[1]), int(nei[0])])

    # =============================================================================================
    # Processing graph label
    # =============================================================================================
    # merge incident edges whose degree is 2
    for k,v in graph.vertices.items():
        if len(v.neighbor_edges)==2 and len([e for e in v.neighbor_edges])==2: 
            graph.merge(v.neighbor_edges)

    # densify edge pixels
    temp_edges = []
    edge_names = []
    output_vertices = []
    graph.vertices = {}
    for e_idx, e in enumerate(graph.edges):
        new_vertices = []
        orientation = []
        for i in range(len(e.vertices)-1):
            dense_segment = get_dense_edge(e.vertices[i],e.vertices[i+1])
            new_vertices.extend(dense_segment)
            orientation_segment = get_orientation_angle(np.array(e.vertices[i+1])-np.array(e.vertices[i]))
            orientation.extend([orientation_segment for _ in range(len(dense_segment))])
        # remove pixels outside the image
        vertex_state_list = [whether_inside(v) for v in new_vertices]
        # do nothing
        if len(np.unique(vertex_state_list)) == 1:
            # update end vertices
            src = graph.add_v(new_vertices[0])
            dst = graph.add_v(new_vertices[-1])
            if f'{src.id}_{dst.id}' not in edge_names:
                src.neighbor_edges.append(e)
                dst.neighbor_edges.append(e)
                edge_names.append(f'{src.id}_{dst.id}')
                edge_names.append(f'{dst.id}_{src.id}')
                e.orientation = orientation
                e.vertices = new_vertices
                e.src, e.dst = src, dst
                temp_edges.append(e)
        # split segments that cross the image edge for multiple times
        else:
            pointer = 0
            current_segment_inside = bool(vertex_state_list[0])
            for v_idx in range(1,len(new_vertices)):
                if not current_segment_inside:
                    pointer = v_idx
                if vertex_state_list[v_idx] != vertex_state_list[v_idx-1] or v_idx==len(new_vertices)-1: # image boundary
                    if current_segment_inside:
                        if v_idx==len(new_vertices)-1 and vertex_state_list[-1]: 
                            current_segment_vertices = new_vertices[pointer:].copy()
                            current_segment_orientation = orientation[pointer:].copy()
                        else:
                            current_segment_vertices = new_vertices[pointer:v_idx].copy()
                            current_segment_orientation = orientation[pointer:v_idx].copy()
                        src = graph.add_v(current_segment_vertices[0])                           
                        dst = graph.add_v(current_segment_vertices[-1])
                        new_e = Edge(src,dst,graph.edge_counter,vertices=current_segment_vertices,orientation=current_segment_orientation)
                        graph.edge_counter += 1
                        if f'{src.id}_{dst.id}' not in edge_names:
                            if new_e not in src.neighbor_edges:src.neighbor_edges.append(new_e)
                            if new_e not in dst.neighbor_edges:dst.neighbor_edges.append(new_e)
                            edge_names.append(f'{src.id}_{dst.id}')
                            edge_names.append(f'{dst.id}_{src.id}')
                            temp_edges.append(new_e)

                    current_segment_inside = not current_segment_inside
    # filter edges
    output_edges = []
    for e in temp_edges:
        src = e.src
        dst = e.dst
        if (len(src.neighbor_edges)<=1 and len(dst.neighbor_edges)<=1) or \
                (not whether_inside(src) and not whether_inside(dst)):
            if e in src.neighbor_edges:src.neighbor_edges.remove(e)
            if e in dst.neighbor_edges:dst.neighbor_edges.remove(e)
        else:
            output_edges.append({'id':e.id,'src':e.src.id,'dst':e.dst.id,'vertices':e.vertices,'orientation':e.orientation})

    # output and vis
    for k,v in graph.vertices.items():
        if len(v.neighbor_edges):
            output_vertices.append({'id':v.id,'x':v.x,'y':v.y,'neighbors':[x.id for x in v.neighbor_edges]})


    with open(f'./dataset/graph/{tile_index}.json','w') as jf:
        json.dump({'edges':output_edges,'vertices':output_vertices},jf)

    vis_map = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE,3)).astype(np.uint8)).convert('RGB')
    draw = ImageDraw.Draw(vis_map)
    p = vis_map.load()
    for e in output_edges:
        for v in e['vertices']:
            try:
                p[int(v[0]),int(v[1])] = (255,255,255)
            except:
                print(v)
                pass

    for k,v in graph.vertices.items():
        if len(v.neighbor_edges) == 1:
            draw.ellipse([v.x-INTER_P_RADIUS,v.y-INTER_P_RADIUS,v.x+INTER_P_RADIUS,v.y+INTER_P_RADIUS],fill='pink')
        elif len(v.neighbor_edges) == 2:
            draw.ellipse([v.x-INTER_P_RADIUS,v.y-INTER_P_RADIUS,v.x+INTER_P_RADIUS,v.y+INTER_P_RADIUS],fill='red')
        elif len(v.neighbor_edges) > 2:
            draw.ellipse([v.x-INTER_P_RADIUS,v.y-INTER_P_RADIUS,v.x+INTER_P_RADIUS,v.y+INTER_P_RADIUS],fill='orange')

    vis_map.save(f'./dataset/vis/{tile_index}.png')

    # =============================================================================================
    # Processing segmentation label
    # =============================================================================================

    # 
    global_mask = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('RGB')
    draw = ImageDraw.Draw(global_mask)
    p = global_mask.load()

    for e in output_edges:
        for i, v in enumerate(e['vertices'][1:]):
            draw.line([e['vertices'][i][0],e['vertices'][i][1],v[0],v[1]],width=SEGMENT_WIDTH,fill=(255,255,255))

    global_mask.save(f'./dataset/segment/{tile_index}.png')

    # 
    global_keypoint_map = Image.fromarray(np.zeros((IMAGE_SIZE,IMAGE_SIZE))).convert('RGB')
    draw = ImageDraw.Draw(global_keypoint_map)
    for v in output_vertices:
        draw.ellipse([v['x']-3,v['y']-3,v['x']+3,v['y']+3],fill='white')

    global_keypoint_map.save(f'./dataset/intersection/{tile_index}.png')

    print(f'tile: {tile_index}/180')
