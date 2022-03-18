"""Functions to help compare PyTorch and TensorFlow Meshes"""
from click import pass_context
import numpy as np
import tensorflow as tf
import torch
import os

TF_OUTPUTS_PATH = r'C:\ML\tf_outputs'
TORCH_OUTPUTS_PATH = r'C:\ML\torch_outputs'

def grab_outputs(output_name):
  tf_output = np.load(os.path.join(TF_OUTPUTS_PATH, output_name + '.npy'))
  torch_output = np.load(os.path.join(TORCH_OUTPUTS_PATH, output_name + '.npy'))

  return tf_output, torch_output

def apply_mask(x, mask):
  x = np.squeeze(x)
  mask = np.squeeze(mask)
  return x[mask == 1]

def compute_ind(x, y):
  ind_x = np.lexsort((x[:, 0], x[:, 1], x[:, 2]), axis=0)
  ind_y = np.lexsort((y[:, 0], y[:, 1], y[:, 2]), axis=0)

  return ind_x, ind_y

def compare_tensors(x, y, ind_x, ind_y):
  x = x[ind_x]
  y = y[ind_y]

  return (np.square(x - y)).mean()

if __name__ == '__main__':
  tf_verts_mask = np.load(os.path.join(TF_OUTPUTS_PATH, "verts_mask.npy"))

  tf_verts, torch_verts = grab_outputs("verts_0")
  tf_verts = apply_mask(tf_verts, tf_verts_mask)
  tf_ind, torch_ind = compute_ind(tf_verts, torch_verts)
  print("Average verts error: ", compare_tensors(tf_verts, torch_verts, tf_ind, torch_ind))

  tf_img_feats, torch_img_feats = grab_outputs("img_feats_0")
  tf_img_feats = apply_mask(tf_img_feats, tf_verts_mask)
  print("Average img_feats error: ", compare_tensors(tf_img_feats, torch_img_feats, tf_ind, torch_ind))

  tf_img_feats_img_feats_bottleneck, torch_img_feats_bottleneck = grab_outputs("img_feats_bottleneck_0")
  tf_img_feats_img_feats_bottleneck = apply_mask(tf_img_feats_img_feats_bottleneck, tf_verts_mask)
  print("Average img_feats bottleneck error: ", compare_tensors(tf_img_feats_img_feats_bottleneck, torch_img_feats_bottleneck, tf_ind, torch_ind))

  tf_vert_feats, torch_vert_feats = grab_outputs("vert_feats_0")
  tf_vert_feats = apply_mask(tf_vert_feats, tf_verts_mask)
  print("Average vert_feats error: ", compare_tensors(tf_vert_feats, torch_vert_feats, tf_ind, torch_ind))

  tf_vert_feats_nopos, torch_vert_feats_nopos = grab_outputs("vert_feats_nopos_0_s0")
  tf_vert_feats_nopos = apply_mask(tf_vert_feats_nopos, tf_verts_mask)
  print("Average vert_feats_nopos_0 error (after 1st GraphConv): ", compare_tensors(tf_vert_feats_nopos, torch_vert_feats_nopos, tf_ind, torch_ind))


  tf_vert_feats_nopos, torch_vert_feats_nopos = grab_outputs("vert_feats_nopos_0_s1")
  tf_vert_feats_nopos = apply_mask(tf_vert_feats_nopos, tf_verts_mask)
  print("Average vert_feats_nopos_1 error (after 2nd GraphConv): ", compare_tensors(tf_vert_feats_nopos, torch_vert_feats_nopos, tf_ind, torch_ind))

  tf_vert_feats_nopos, torch_vert_feats_nopos = grab_outputs("vert_feats_nopos_0_s2")
  tf_vert_feats_nopos = apply_mask(tf_vert_feats_nopos, tf_verts_mask)
  print("Average vert_feats_nopos_2 error (after 3rd GraphConv): ", compare_tensors(tf_vert_feats_nopos, torch_vert_feats_nopos, tf_ind, torch_ind))

  tf_deform, torch_deform = grab_outputs("deform_0")
  tf_deform = apply_mask(tf_deform, tf_verts_mask)
  print("deform differences: ", compare_tensors(tf_deform, torch_deform, tf_ind, torch_ind))

  