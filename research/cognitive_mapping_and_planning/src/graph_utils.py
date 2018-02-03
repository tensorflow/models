# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Various function to manipulate graphs for computing distances.
"""
import skimage.morphology
import numpy as np
import networkx as nx
import itertools
import logging
from datasets.nav_env import get_path_ids
import graph_tool as gt
import graph_tool.topology
import graph_tool.generation
import src.utils as utils

# Compute shortest path from all nodes to or from all source nodes
def get_distance_node_list(gtG, source_nodes, direction, weights=None):
  gtG_ = gt.Graph(gtG)
  v = gtG_.add_vertex()

  if weights is not None:
    weights = gtG_.edge_properties[weights]

  for s in source_nodes:
    e = gtG_.add_edge(s, int(v))
    if weights is not None:
      weights[e] = 0.

  if direction == 'to':
    dist = gt.topology.shortest_distance(
        gt.GraphView(gtG_, reversed=True), source=gtG_.vertex(int(v)),
        target=None, weights=weights)
  elif direction == 'from':
    dist = gt.topology.shortest_distance(
        gt.GraphView(gtG_, reversed=False), source=gtG_.vertex(int(v)),
        target=None, weights=weights)
  dist = np.array(dist.get_array())
  dist = dist[:-1]
  if weights is None:
    dist = dist-1
  return dist

# Functions for semantically labelling nodes in the traversal graph.
def generate_lattice(sz_x, sz_y):
  """Generates a lattice with sz_x vertices along x and sz_y vertices along y
  direction Each of these vertices is step_size distance apart. Origin is at
  (0,0).  """
  g = gt.generation.lattice([sz_x, sz_y])
  x, y = np.meshgrid(np.arange(sz_x), np.arange(sz_y))
  x = np.reshape(x, [-1,1]); y = np.reshape(y, [-1,1]);
  nodes = np.concatenate((x,y), axis=1)
  return g, nodes

def add_diagonal_edges(g, nodes, sz_x, sz_y, edge_len):
  offset = [sz_x+1, sz_x-1]
  for o in offset:
    s = np.arange(nodes.shape[0]-o-1)
    t = s + o
    ind = np.all(np.abs(nodes[s,:] - nodes[t,:]) == np.array([[1,1]]), axis=1)
    s = s[ind][:,np.newaxis]
    t = t[ind][:,np.newaxis]
    st = np.concatenate((s,t), axis=1)
    for i in range(st.shape[0]):
      e = g.add_edge(st[i,0], st[i,1], add_missing=False)
      g.ep['wts'][e] = edge_len

def convert_traversible_to_graph(traversible, ff_cost=1., fo_cost=1.,
                                 oo_cost=1., connectivity=4):
  assert(connectivity == 4 or connectivity == 8)

  sz_x = traversible.shape[1]
  sz_y = traversible.shape[0]
  g, nodes = generate_lattice(sz_x, sz_y)

  # Assign costs.
  edge_wts = g.new_edge_property('float')
  g.edge_properties['wts'] = edge_wts
  wts = np.ones(g.num_edges(), dtype=np.float32)
  edge_wts.get_array()[:] = wts

  if connectivity == 8:
    add_diagonal_edges(g, nodes, sz_x, sz_y, np.sqrt(2.))

  se = np.array([[int(e.source()), int(e.target())] for e in g.edges()])
  s_xy = nodes[se[:,0]]
  t_xy = nodes[se[:,1]]
  s_t = np.ravel_multi_index((s_xy[:,1], s_xy[:,0]), traversible.shape)
  t_t = np.ravel_multi_index((t_xy[:,1], t_xy[:,0]), traversible.shape)
  s_t = traversible.ravel()[s_t]
  t_t = traversible.ravel()[t_t]

  wts = np.zeros(g.num_edges(), dtype=np.float32)
  wts[np.logical_and(s_t == True, t_t == True)] = ff_cost
  wts[np.logical_and(s_t == False, t_t == False)] = oo_cost
  wts[np.logical_xor(s_t, t_t)] = fo_cost

  edge_wts = g.edge_properties['wts']
  for i, e in enumerate(g.edges()):
    edge_wts[e] = edge_wts[e] * wts[i]
  # d = edge_wts.get_array()*1.
  # edge_wts.get_array()[:] = d*wts
  return g, nodes

def label_nodes_with_class(nodes_xyt, class_maps, pix):
  """
  Returns:
    class_maps__: one-hot class_map for each class.
    node_class_label: one-hot class_map for each class, nodes_xyt.shape[0] x n_classes
  """
  # Assign each pixel to a node.
  selem = skimage.morphology.disk(pix)
  class_maps_ = class_maps*1.
  for i in range(class_maps.shape[2]):
    class_maps_[:,:,i] = skimage.morphology.dilation(class_maps[:,:,i]*1, selem)
  class_maps__ = np.argmax(class_maps_, axis=2)
  class_maps__[np.max(class_maps_, axis=2) == 0] = -1

  # For each node pick out the label from this class map.
  x = np.round(nodes_xyt[:,[0]]).astype(np.int32)
  y = np.round(nodes_xyt[:,[1]]).astype(np.int32)
  ind = np.ravel_multi_index((y,x), class_maps__.shape)
  node_class_label = class_maps__.ravel()[ind][:,0]

  # Convert to one hot versions.
  class_maps_one_hot = np.zeros(class_maps.shape, dtype=np.bool)
  node_class_label_one_hot = np.zeros((node_class_label.shape[0], class_maps.shape[2]), dtype=np.bool)
  for i in range(class_maps.shape[2]):
    class_maps_one_hot[:,:,i] = class_maps__ == i
    node_class_label_one_hot[:,i] = node_class_label == i
  return class_maps_one_hot, node_class_label_one_hot

def label_nodes_with_class_geodesic(nodes_xyt, class_maps, pix, traversible,
                                    ff_cost=1., fo_cost=1., oo_cost=1.,
                                    connectivity=4):
  """Labels nodes in nodes_xyt with class labels using geodesic distance as
  defined by traversible from class_maps.
  Inputs:
    nodes_xyt
    class_maps: counts for each class.
    pix: distance threshold to consider close enough to target.
    traversible: binary map of whether traversible or not.
  Output:
    labels: For each node in nodes_xyt returns a label of the class or -1 is
    unlabelled.
  """
  g, nodes = convert_traversible_to_graph(traversible, ff_cost=ff_cost,
                                          fo_cost=fo_cost, oo_cost=oo_cost,
                                          connectivity=connectivity)

  class_dist = np.zeros_like(class_maps*1.)
  n_classes = class_maps.shape[2]
  if False:
    # Assign each pixel to a class based on number of points.
    selem = skimage.morphology.disk(pix)
    class_maps_ = class_maps*1.
    class_maps__ = np.argmax(class_maps_, axis=2)
    class_maps__[np.max(class_maps_, axis=2) == 0] = -1

  # Label nodes with classes.
  for i in range(n_classes):
    # class_node_ids = np.where(class_maps__.ravel() == i)[0]
    class_node_ids = np.where(class_maps[:,:,i].ravel() > 0)[0]
    dist_i = get_distance_node_list(g, class_node_ids, 'to', weights='wts')
    class_dist[:,:,i] = np.reshape(dist_i, class_dist[:,:,i].shape)
  class_map_geodesic = (class_dist <= pix)
  class_map_geodesic = np.reshape(class_map_geodesic, [-1, n_classes])

  # For each node pick out the label from this class map.
  x = np.round(nodes_xyt[:,[0]]).astype(np.int32)
  y = np.round(nodes_xyt[:,[1]]).astype(np.int32)
  ind = np.ravel_multi_index((y,x), class_dist[:,:,0].shape)
  node_class_label = class_map_geodesic[ind[:,0],:]
  class_map_geodesic = class_dist <= pix
  return class_map_geodesic, node_class_label

def _get_next_nodes_undirected(n, sc, n_ori):
  nodes_to_add = []
  nodes_to_validate = []
  (p, q, r) = n
  nodes_to_add.append((n, (p, q, r), 0))
  if n_ori == 4:
    for _ in [1, 2, 3, 4]:
      if _ == 1:
        v = (p - sc, q, r)
      elif _ == 2:
        v = (p + sc, q, r)
      elif _ == 3:
        v = (p, q - sc, r)
      elif _ == 4:
        v = (p, q + sc, r)
      nodes_to_validate.append((n, v, _))
  return nodes_to_add, nodes_to_validate

def _get_next_nodes(n, sc, n_ori):
  nodes_to_add = []
  nodes_to_validate = []
  (p, q, r) = n
  for r_, a_ in zip([-1, 0, 1], [1, 0, 2]):
    nodes_to_add.append((n, (p, q, np.mod(r+r_, n_ori)), a_))

  if n_ori == 6:
    if r == 0:
      v = (p + sc, q, r)
    elif r == 1:
      v = (p + sc, q + sc, r)
    elif r == 2:
      v = (p, q + sc, r)
    elif r == 3:
      v = (p - sc, q, r)
    elif r == 4:
      v = (p - sc, q - sc, r)
    elif r == 5:
      v = (p, q - sc, r)
  elif n_ori == 4:
    if r == 0:
      v = (p + sc, q, r)
    elif r == 1:
      v = (p, q + sc, r)
    elif r == 2:
      v = (p - sc, q, r)
    elif r == 3:
      v = (p, q - sc, r)
  nodes_to_validate.append((n,v,3))

  return nodes_to_add, nodes_to_validate

def generate_graph(valid_fn_vec=None, sc=1., n_ori=6,
                   starting_location=(0, 0, 0), vis=False, directed=True):
  timer = utils.Timer()
  timer.tic()
  if directed: G = nx.DiGraph(directed=True)
  else: G = nx.Graph()
  G.add_node(starting_location)
  new_nodes = G.nodes()
  while len(new_nodes) != 0:
    nodes_to_add = []
    nodes_to_validate = []
    for n in new_nodes:
      if directed:
        na, nv = _get_next_nodes(n, sc, n_ori)
      else:
        na, nv = _get_next_nodes_undirected(n, sc, n_ori)
      nodes_to_add = nodes_to_add + na
      if valid_fn_vec is not None:
        nodes_to_validate = nodes_to_validate + nv
      else:
        node_to_add = nodes_to_add + nv

    # Validate nodes.
    vs = [_[1] for _ in nodes_to_validate]
    valids = valid_fn_vec(vs)

    for nva, valid in zip(nodes_to_validate, valids):
      if valid:
        nodes_to_add.append(nva)

    new_nodes = []
    for n,v,a in nodes_to_add:
      if not G.has_node(v):
        new_nodes.append(v)
      G.add_edge(n, v, action=a)

  timer.toc(average=True, log_at=1, log_str='src.graph_utils.generate_graph')
  return (G)

def vis_G(G, ax, vertex_color='r', edge_color='b', r=None):
  if edge_color is not None:
    for e in G.edges():
      XYT = zip(*e)
      x = XYT[-3]
      y = XYT[-2]
      t = XYT[-1]
      if r is None or t[0] == r:
        ax.plot(x, y, edge_color)
  if vertex_color is not None:
    XYT = zip(*G.nodes())
    x = XYT[-3]
    y = XYT[-2]
    t = XYT[-1]
    ax.plot(x, y, vertex_color + '.')

def convert_to_graph_tool(G):
  timer = utils.Timer()
  timer.tic()
  gtG = gt.Graph(directed=G.is_directed())
  gtG.ep['action'] = gtG.new_edge_property('int')

  nodes_list = G.nodes()
  nodes_array = np.array(nodes_list)

  nodes_id = np.zeros((nodes_array.shape[0],), dtype=np.int64)

  for i in range(nodes_array.shape[0]):
    v = gtG.add_vertex()
    nodes_id[i] = int(v)

  # d = {key: value for (key, value) in zip(nodes_list, nodes_id)}
  d = dict(itertools.izip(nodes_list, nodes_id))

  for src, dst, data in G.edges_iter(data=True):
    e = gtG.add_edge(d[src], d[dst])
    gtG.ep['action'][e] = data['action']
  nodes_to_id = d
  timer.toc(average=True, log_at=1, log_str='src.graph_utils.convert_to_graph_tool')
  return gtG, nodes_array, nodes_to_id


def _rejection_sampling(rng, sampling_d, target_d, bins, hardness, M):
  bin_ind = np.digitize(hardness, bins)-1
  i = 0
  ratio = target_d[bin_ind] / (M*sampling_d[bin_ind])
  while i < ratio.size and rng.rand() > ratio[i]:
    i = i+1
  return i

def heuristic_fn_vec(n1, n2, n_ori, step_size):
  # n1 is a vector and n2 is a single point.
  dx = (n1[:,0] - n2[0,0])/step_size
  dy = (n1[:,1] - n2[0,1])/step_size
  dt = n1[:,2] - n2[0,2]
  dt = np.mod(dt, n_ori)
  dt = np.minimum(dt, n_ori-dt)

  if n_ori == 6:
    if dx*dy > 0:
      d = np.maximum(np.abs(dx), np.abs(dy))
    else:
      d = np.abs(dy-dx)
  elif n_ori == 4:
    d = np.abs(dx) + np.abs(dy)

  return (d + dt).reshape((-1,1))

def get_hardness_distribution(gtG, max_dist, min_dist, rng, trials, bins, nodes,
                              n_ori, step_size):
  heuristic_fn = lambda node_ids, node_id: \
    heuristic_fn_vec(nodes[node_ids, :], nodes[[node_id], :], n_ori, step_size)
  num_nodes = gtG.num_vertices()
  gt_dists = []; h_dists = [];
  for i in range(trials):
    end_node_id = rng.choice(num_nodes)
    gt_dist = gt.topology.shortest_distance(gt.GraphView(gtG, reversed=True),
                                            source=gtG.vertex(end_node_id),
                                            target=None, max_dist=max_dist)
    gt_dist = np.array(gt_dist.get_array())
    ind = np.where(np.logical_and(gt_dist <= max_dist, gt_dist >= min_dist))[0]
    gt_dist = gt_dist[ind]
    h_dist = heuristic_fn(ind, end_node_id)[:,0]
    gt_dists.append(gt_dist)
    h_dists.append(h_dist)
  gt_dists = np.concatenate(gt_dists)
  h_dists = np.concatenate(h_dists)
  hardness = 1. - h_dists*1./gt_dists
  hist, _ = np.histogram(hardness, bins)
  hist = hist.astype(np.float64)
  hist = hist / np.sum(hist)
  return hist

def rng_next_goal_rejection_sampling(start_node_ids, batch_size, gtG, rng,
                                     max_dist, min_dist, max_dist_to_compute,
                                     sampling_d, target_d,
                                     nodes, n_ori, step_size, bins, M):
  sample_start_nodes = start_node_ids is None
  dists = []; pred_maps = []; end_node_ids = []; start_node_ids_ = [];
  hardnesss = []; gt_dists = [];
  num_nodes = gtG.num_vertices()
  for i in range(batch_size):
    done = False
    while not done:
      if sample_start_nodes:
        start_node_id = rng.choice(num_nodes)
      else:
        start_node_id = start_node_ids[i]

      gt_dist = gt.topology.shortest_distance(
          gt.GraphView(gtG, reversed=False), source=start_node_id, target=None,
          max_dist=max_dist)
      gt_dist = np.array(gt_dist.get_array())
      ind = np.where(np.logical_and(gt_dist <= max_dist, gt_dist >= min_dist))[0]
      ind = rng.permutation(ind)
      gt_dist = gt_dist[ind]*1.
      h_dist = heuristic_fn_vec(nodes[ind, :], nodes[[start_node_id], :],
                                n_ori, step_size)[:,0]
      hardness = 1. - h_dist / gt_dist
      sampled_ind = _rejection_sampling(rng, sampling_d, target_d, bins,
                                        hardness, M)
      if sampled_ind < ind.size:
        # print sampled_ind
        end_node_id = ind[sampled_ind]
        hardness = hardness[sampled_ind]
        gt_dist = gt_dist[sampled_ind]
        done = True

    # Compute distance from end node to all nodes, to return.
    dist, pred_map = gt.topology.shortest_distance(
        gt.GraphView(gtG, reversed=True), source=end_node_id, target=None,
        max_dist=max_dist_to_compute, pred_map=True)
    dist = np.array(dist.get_array())
    pred_map = np.array(pred_map.get_array())

    hardnesss.append(hardness); dists.append(dist); pred_maps.append(pred_map);
    start_node_ids_.append(start_node_id); end_node_ids.append(end_node_id);
    gt_dists.append(gt_dist);
    paths = None
  return start_node_ids_, end_node_ids, dists, pred_maps, paths, hardnesss, gt_dists


def rng_next_goal(start_node_ids, batch_size, gtG, rng, max_dist,
                  max_dist_to_compute, node_room_ids, nodes=None,
                  compute_path=False, dists_from_start_node=None):
  # Compute the distance field from the starting location, and then pick a
  # destination in another room if possible otherwise anywhere outside this
  # room.
  dists = []; pred_maps = []; paths = []; end_node_ids = [];
  for i in range(batch_size):
    room_id = node_room_ids[start_node_ids[i]]
    # Compute distances.
    if dists_from_start_node == None:
      dist, pred_map = gt.topology.shortest_distance(
        gt.GraphView(gtG, reversed=False), source=gtG.vertex(start_node_ids[i]),
        target=None, max_dist=max_dist_to_compute, pred_map=True)
      dist = np.array(dist.get_array())
    else:
      dist = dists_from_start_node[i]

    # Randomly sample nodes which are within max_dist.
    near_ids = dist <= max_dist
    near_ids = near_ids[:, np.newaxis]
    # Check to see if there is a non-negative node which is close enough.
    non_same_room_ids = node_room_ids != room_id
    non_hallway_ids = node_room_ids != -1
    good1_ids = np.logical_and(near_ids, np.logical_and(non_same_room_ids, non_hallway_ids))
    good2_ids = np.logical_and(near_ids, non_hallway_ids)
    good3_ids = near_ids
    if np.any(good1_ids):
      end_node_id = rng.choice(np.where(good1_ids)[0])
    elif np.any(good2_ids):
      end_node_id = rng.choice(np.where(good2_ids)[0])
    elif np.any(good3_ids):
      end_node_id = rng.choice(np.where(good3_ids)[0])
    else:
      logging.error('Did not find any good nodes.')

    # Compute distance to this new goal for doing distance queries.
    dist, pred_map = gt.topology.shortest_distance(
        gt.GraphView(gtG, reversed=True), source=gtG.vertex(end_node_id),
        target=None, max_dist=max_dist_to_compute, pred_map=True)
    dist = np.array(dist.get_array())
    pred_map = np.array(pred_map.get_array())

    dists.append(dist)
    pred_maps.append(pred_map)
    end_node_ids.append(end_node_id)

    path = None
    if compute_path:
      path = get_path_ids(start_node_ids[i], end_node_ids[i], pred_map)
    paths.append(path)

  return start_node_ids, end_node_ids, dists, pred_maps, paths


def rng_room_to_room(batch_size, gtG, rng, max_dist, max_dist_to_compute,
                     node_room_ids, nodes=None, compute_path=False):
  # Sample one of the rooms, compute the distance field. Pick a destination in
  # another room if possible otherwise anywhere outside this room.
  dists = []; pred_maps = []; paths = []; start_node_ids = []; end_node_ids = [];
  room_ids = np.unique(node_room_ids[node_room_ids[:,0] >= 0, 0])
  for i in range(batch_size):
    room_id = rng.choice(room_ids)
    end_node_id = rng.choice(np.where(node_room_ids[:,0] == room_id)[0])
    end_node_ids.append(end_node_id)

    # Compute distances.
    dist, pred_map = gt.topology.shortest_distance(
        gt.GraphView(gtG, reversed=True), source=gtG.vertex(end_node_id),
        target=None, max_dist=max_dist_to_compute, pred_map=True)
    dist = np.array(dist.get_array())
    pred_map = np.array(pred_map.get_array())
    dists.append(dist)
    pred_maps.append(pred_map)

    # Randomly sample nodes which are within max_dist.
    near_ids = dist <= max_dist
    near_ids = near_ids[:, np.newaxis]

    # Check to see if there is a non-negative node which is close enough.
    non_same_room_ids = node_room_ids != room_id
    non_hallway_ids = node_room_ids != -1
    good1_ids = np.logical_and(near_ids, np.logical_and(non_same_room_ids, non_hallway_ids))
    good2_ids = np.logical_and(near_ids, non_hallway_ids)
    good3_ids = near_ids
    if np.any(good1_ids):
      start_node_id = rng.choice(np.where(good1_ids)[0])
    elif np.any(good2_ids):
      start_node_id = rng.choice(np.where(good2_ids)[0])
    elif np.any(good3_ids):
      start_node_id = rng.choice(np.where(good3_ids)[0])
    else:
      logging.error('Did not find any good nodes.')

    start_node_ids.append(start_node_id)

    path = None
    if compute_path:
      path = get_path_ids(start_node_ids[i], end_node_ids[i], pred_map)
    paths.append(path)

  return start_node_ids, end_node_ids, dists, pred_maps, paths


def rng_target_dist_field(batch_size, gtG, rng, max_dist, max_dist_to_compute,
                          nodes=None, compute_path=False):
  # Sample a single node, compute distance to all nodes less than max_dist,
  # sample nodes which are a particular distance away.
  dists = []; pred_maps = []; paths = []; start_node_ids = []
  end_node_ids = rng.choice(gtG.num_vertices(), size=(batch_size,),
                            replace=False).tolist()

  for i in range(batch_size):
    dist, pred_map = gt.topology.shortest_distance(
        gt.GraphView(gtG, reversed=True), source=gtG.vertex(end_node_ids[i]),
        target=None, max_dist=max_dist_to_compute, pred_map=True)
    dist = np.array(dist.get_array())
    pred_map = np.array(pred_map.get_array())
    dists.append(dist)
    pred_maps.append(pred_map)

    # Randomly sample nodes which are withing max_dist
    near_ids = np.where(dist <= max_dist)[0]
    start_node_id = rng.choice(near_ids, size=(1,), replace=False)[0]
    start_node_ids.append(start_node_id)

    path = None
    if compute_path:
      path = get_path_ids(start_node_ids[i], end_node_ids[i], pred_map)
    paths.append(path)

  return start_node_ids, end_node_ids, dists, pred_maps, paths
