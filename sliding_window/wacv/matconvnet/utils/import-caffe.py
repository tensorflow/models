#! /usr/bin/python
# file: import-caffe.py
# brief: Caffe importer for DagNN and SimpleNN
# author: Karel Lenc and Andrea Vedaldi

# Requires Google Protobuf for Python and SciPy

import sys
import os
import argparse
import code
import re
import numpy as np
from math import floor, ceil
import numpy
from numpy import array
import scipy
import scipy.io
import scipy.misc
import google.protobuf.text_format
from ast import literal_eval as make_tuple
from layers import *

# --------------------------------------------------------------------
#                                                  Check NumPy version
# --------------------------------------------------------------------

def versiontuple(version):
  return tuple(map(int, (version.split("."))))

min_numpy_version = "1.7.0"
if versiontuple(numpy.version.version) < versiontuple(min_numpy_version):
  print 'Unsupported numpy version ({}), must be >= {}'.format(numpy.version.version,
    min_numpy_version)
  sys.exit(0)

# --------------------------------------------------------------------
#                                                     Helper functions
# --------------------------------------------------------------------

def find(seq, name):
  for item in seq:
    if item.name == name:
      return item
  return None

def blobproto_to_array(blob):
  """Convert a Caffe Blob to a numpy array.

It also reverses the order of all dimensions to [width, height,
channels, instance].
"""
  dims = []
  if hasattr(blob, 'shape'):
    dims = tolist(blob.shape.dim)
  if not dims:
    dims = [blob.num, blob.channels, blob.height, blob.width]
  return np.array(blob.data,dtype='float32').reshape(dims).transpose()

def dict_to_struct_array(d):
  if not d:
    return np.zeros((0,))
  dt=[(x,object) for x in d.keys()]
  y = np.empty((1,),dtype=dt)
  for x in d.keys():
    y[x][0] = d[x]
  return y

def tolist(x):
  "Convert x to a Python list. x can be a Protobuf container, a list or tuple, or scalar"
  if isinstance(x,google.protobuf.internal.containers.RepeatedScalarFieldContainer):
    return [z for z in x]
  elif isinstance(x, (list,tuple)):
    return [z for z in x]
  else:
    return [x]

def escape(name):
  return name.replace('-','_')

# --------------------------------------------------------------------
#                                                        Parse options
# --------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Convert a Caffe CNN into a MATLAB structure.')
parser.add_argument('caffe_proto',
                    type=argparse.FileType('rb'),
                    help='The Caffe CNN parameter file (ASCII .proto)')
parser.add_argument('--caffe-data',
                    type=argparse.FileType('rb'),
                    help='The Caffe CNN data file (binary .proto)')
parser.add_argument('output',
                    type=argparse.FileType('w'),
                    help='Output MATLAB file')
parser.add_argument('--full-image-size',
                    type=str,
                    nargs='?',
                    default=None,
                    help='Size of the full image')
parser.add_argument('--average-image',
                    type=argparse.FileType('rb'),
                    nargs='?',
                    help='Average image')
parser.add_argument('--average-value',
                    type=str,
                    nargs='?',
                    default=None,
                    help='Average image value')
parser.add_argument('--synsets',
                    type=argparse.FileType('r'),
                    nargs='?',
                    help='Synset file (ASCII)')
parser.add_argument('--class-names',
                    type=str,
                    nargs='?',
                    help='Class names')
parser.add_argument('--caffe-variant',
                    type=str,
                    nargs='?',
                    default='caffe',
                    help='Variant of Caffe software (use ? to get a list)')
parser.add_argument('--transpose',
                    dest='transpose',
                    action='store_true',
                    help='Transpose CNN in a sane MATLAB format')
parser.add_argument('--no-transpose',
                    dest='transpose',
                    action='store_false',
                    help='Do not transpose CNN')
parser.add_argument('--color-format',
                    dest='color_format',
                    default='bgr',
                    action='store',
                    help='Set the color format used by the network: ''rgb'' or ''bgr'' (default)')
parser.add_argument('--preproc',
                    type=str,
                    nargs='?',
                    default='caffe',
                    help='Variant of image preprocessing to use (use ? to get a list)')
parser.add_argument('--simplify',
                    dest='simplify',
                    action='store_true',
                    help='Apply simplifications')
parser.add_argument('--no-simplify',
                    dest='simplify',
                    action='store_false',
                    help='Do not apply simplifications')
parser.add_argument('--remove-dropout',
                    dest='remove_dropout',
                    action='store_true',
                    help='Remove dropout layers')
parser.add_argument('--no-remove-dropout',
                    dest='remove_dropout',
                    action='store_false',
                    help='Do not remove dropout layers')
parser.add_argument('--remove-loss',
                    dest='remove_loss',
                    action='store_true',
                    help='Remove loss layers')
parser.add_argument('--no-remove-loss',
                    dest='remove_loss',
                    action='store_false',
                    help='Do not remove loss layers')
parser.add_argument('--append-softmax',
                    dest='append_softmax',
                    action='append',
                    default=[],
                    help='Add a softmax layer after the specified layer')
parser.add_argument('--output-format',
                    dest='output_format',
                    default='dagnn',
                    help='Either ''dagnn'' or ''simplenn''')

parser.set_defaults(transpose=True)
parser.set_defaults(remove_dropout=False)
parser.set_defaults(remove_loss=False)
parser.set_defaults(simplify=True)
args = parser.parse_args()

print 'Caffe varaint set to', args.caffe_variant
if args.caffe_variant == 'vgg-caffe':
  import proto.vgg_caffe_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe-old':
  import proto.caffe_old_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe':
  import proto.caffe_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe_0115':
  import proto.caffe_0115_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe_6e3916':
  import proto.caffe_6e3916_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe_b590f1d':
  import proto.caffe_b590f1d_pb2 as caffe_pb2
elif args.caffe_variant == 'caffe_fastrcnn':
  import proto.caffe_fastrcnn_pb2 as caffe_pb2
elif args.caffe_variant == '?':
  print 'Supported variants: caffe, vgg-caffe, caffe-old, caffe_0115, caffe_6e3916, caffe_b590f1d, caffe_fastrcnn'
  sys.exit(0)
else:
  print 'Unknown Caffe variant', args.caffe_variant
  sys.exit(1)

if args.preproc == '?':
  print 'Preprocessing variants: caffe, vgg, fcn'
  sys.exit(0)
if args.preproc not in ['caffe', 'vgg-caffe', 'fcn']:
  print 'Unknown preprocessing variant', args.preproc
  sys.exit(1)

# --------------------------------------------------------------------
#                                                     Helper functions
# --------------------------------------------------------------------

def keyboard(banner=None):
    ''' Function that mimics the matlab keyboard command '''
    # use exception trick to pick up the current frame
    try:
        raise None
    except:
        frame = sys.exc_info()[2].tb_frame.f_back
    print "# Use quit() to exit :) Happy debugging!"
    # evaluate commands in current namespace
    namespace = frame.f_globals.copy()
    namespace.update(frame.f_locals)
    try:
        code.interact(banner=banner, local=namespace)
    except SystemExit:
        return

def bilinear_interpolate(im, x, y):
  x = np.asarray(x)
  y = np.asarray(y)

  x0 = np.floor(x).astype(int)
  x1 = x0 + 1
  y0 = np.floor(y).astype(int)
  y1 = y0 + 1

  x0 = np.clip(x0, 0, im.shape[1]-1);
  x1 = np.clip(x1, 0, im.shape[1]-1);
  y0 = np.clip(y0, 0, im.shape[0]-1);
  y1 = np.clip(y1, 0, im.shape[0]-1);

  Ia = im[ y0, x0 ]
  Ib = im[ y1, x0 ]
  Ic = im[ y0, x1 ]
  Id = im[ y1, x1 ]

  wa = (1-x+x0) * (1-y+y0)
  wb = (1-x+x0) * (y-y0)
  wc = (x-x0) * (1-y+y0)
  wd = (x-x0) * (y-y0)

  wa = wa.reshape(x.shape[0], x.shape[1], 1)
  wb = wb.reshape(x.shape[0], x.shape[1], 1)
  wc = wc.reshape(x.shape[0], x.shape[1], 1)
  wd = wd.reshape(x.shape[0], x.shape[1], 1)

  return wa*Ia + wb*Ib + wc*Ic + wd*Id

# Get the parameters for a layer from Caffe's proto entries
def getopts(layer, name):
  if hasattr(layer, name):
    return getattr(layer, name)
  else:
    # Older Caffe proto formats did not have sub-structures for layer
    # specific parameters but mixed everything up! This falls back to
    # that situation when fetching the parameters.
    return layer

# --------------------------------------------------------------------
#                                                   Load average image
# --------------------------------------------------------------------

average_image = None
resize_average_image = False
if args.average_image:
  print 'Loading average image from {}'.format(args.average_image.name)
  resize_average_image = True # in case different from data size
  avgim_nm, avgim_ext = os.path.splitext(args.average_image.name)
  if avgim_ext == '.binaryproto':
    blob=caffe_pb2.BlobProto()
    blob.MergeFromString(args.average_image.read())
    average_image = blobproto_to_array(blob).astype('float32')
    average_image = np.squeeze(average_image,3)
    if args.transpose and average_image is not None:
      average_image = average_image.transpose([1,0,2])
      average_image = average_image[:,:,: : -1] # to RGB
  elif avgim_ext == '.mat':
    avgim_data = scipy.io.loadmat(args.average_image)
    average_image = avgim_data['mean_img']
  else:
    print 'Unsupported average image format {}'.format(avgim_ext)

if args.average_value:
  rgb = make_tuple(args.average_value)
  print 'Using average image value', rgb
  # this will be resized later to a constant image
  average_image = np.array(rgb,dtype=float).reshape(1,1,3,order='F')
  resize_average_image = False

# --------------------------------------------------------------------
#                                      Load ImageNet synseths (if any)
# --------------------------------------------------------------------

synsets_wnid=None
synsets_name=None

if args.synsets:
  print 'Loading synsets from {}'.format(args.synsets.name)
  r=re.compile('(?P<wnid>n[0-9]{8}?) (?P<name>.*)')
  synsets_wnid=[]
  synsets_name=[]
  for line in args.synsets:
    match = r.match(line)
    synsets_wnid.append(match.group('wnid'))
    synsets_name.append(match.group('name'))

if args.class_names:
  synsets_wnid=list(make_tuple(args.class_names))
  synsets_name=synsets_wnid

# --------------------------------------------------------------------
#                                                          Load layers
# --------------------------------------------------------------------

# Caffe stores the network structure and data into two different files
# We load them both and merge them into a single MATLAB structure

net=caffe_pb2.NetParameter()
data=caffe_pb2.NetParameter()

print 'Loading Caffe CNN structure from {}'.format(args.caffe_proto.name)
google.protobuf.text_format.Merge(args.caffe_proto.read(), net)

if args.caffe_data:
  print 'Loading Caffe CNN parameters from {}'.format(args.caffe_data.name)
  data.MergeFromString(args.caffe_data.read())

# --------------------------------------------------------------------
#                                   Read layers in a CaffeModel object
# --------------------------------------------------------------------

if args.caffe_variant in ['caffe_b590f1d', 'caffe_fastrcnn']:
  layers_list = net.layer
  data_layers_list = data.layer
else:
  layers_list = net.layers
  data_layers_list = data.layers

print 'Converting {} layers'.format(len(layers_list))

cmodel = CaffeModel()
for layer in layers_list:

  # Depending on how old the proto-buf, the top and bottom parameters
  # are found at a different level than the others
  top = layer.top
  bottom = layer.bottom
  if args.caffe_variant in ['vgg-caffe', 'caffe-old']:
    layer = layer.layer

  # get the type of layer
  # depending on the Caffe variant, this is a string or a numeric
  # ID, which we convert back to a string
  ltype = layer.type
  if not isinstance(ltype, basestring): ltype = layers_type[ltype]
  print 'Added layer \'{}\' ({})'.format(ltype, layer.name)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if ltype in ['conv', 'deconvolution', 'Convolution', 'Deconvolution']:
    opts = getopts(layer, 'convolution_param')
    if hasattr(opts, 'kernelsize'):
      kernel_size = opts.kernelsize
    else:
      kernel_size = opts.kernel_size
    if hasattr(opts, 'bias_term'):
      bias_term = opts.bias_term
    else:
      bias_term = True
    if hasattr(opts, 'dilation'):
      dilation = opts.dilation
    else:
      dilation = 1
    if ltype in ['conv', 'Convolution']:
      clayer = CaffeConv(layer.name, bottom, top,
                         kernel_size = tolist(kernel_size),
                         bias_term = bias_term,
                         num_output = opts.num_output,
                         group = opts.group,
                         dilation = dilation,
                         stride = tolist(opts.stride),
                         pad = tolist(opts.pad))
    else:
      clayer = CaffeDeconvolution(layer.name, bottom, top,
                                  kernel_size = tolist(kernel_size),
                                  bias_term = bias_term,
                                  num_output = opts.num_output,
                                  group = opts.group,
                                  dilation = dilation,
                                  stride = tolist(opts.stride),
                                  pad = tolist(opts.pad))

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['innerproduct', 'inner_product', 'InnerProduct']:
    opts = getopts(layer, 'inner_product_param')
    if hasattr(opts, 'bias_term'):
      bias_term = opts.bias_term
    else:
      bias_term = True
    if hasattr(opts, 'axis'):
      axis = opts.axis
    else:
      axis = 1
    clayer = CaffeInnerProduct(layer.name, bottom, top,
                               num_output = opts.num_output,
                               bias_term = bias_term,
                               axis = axis)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['relu', 'ReLU']:
    clayer = CaffeReLU(layer.name, bottom, top)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['crop', 'Crop']:
    clayer = CaffeCrop(layer.name, bottom, top)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['lrn', 'LRN']:
    opts = getopts(layer, 'lrn_param')
    local_size = float(opts.local_size)
    alpha = float(opts.alpha)
    beta = float(opts.beta)
    kappa = opts.k if hasattr(opts,'k') else 1.
    regions = ['across_channels', 'within_channel']
    if hasattr(opts, 'norm_region'):
      norm_region = opts.norm_region
    else:
      norm_region = 0
    clayer = CaffeLRN(layer.name, bottom, top,
                      local_size = local_size,
                      alpha = alpha,
                      beta = beta,
                      norm_region = regions[norm_region],
                      kappa = kappa)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['pool', 'Pooling']:
    opts = getopts(layer, 'pooling_param')
    if hasattr(layer, 'kernelsize'):
      kernel_size = opts.kernelsize
    else:
      kernel_size = opts.kernel_size
    clayer = CaffePooling(layer.name, bottom, top,
                          method = ['max', 'avg'][opts.pool],
                          pad = tolist(opts.pad),
                          kernel_size = tolist(kernel_size),
                          stride = tolist(opts.stride))

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['dropout', 'Dropout']:
    opts = getopts(layer, 'dropout_param')
    clayer = CaffeDropout(layer.name, bottom, top,
                          opts.dropout_ratio)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['softmax', 'Softmax']:
    clayer = CaffeSoftMax(layer.name, bottom, top)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['softmax_loss', 'SoftmaxLoss']:
    clayer = CaffeSoftMaxLoss(layer.name, bottom, top)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['concat', 'Concat']:
    opts = getopts(layer, 'concat_param')
    clayer = CaffeConcat(layer.name, bottom, top,
                         3 - opts.concat_dim) # todo: depreceted in recent Caffes

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['Scale']:
    opts = getopts(layer, 'scale_param')
    clayer = CaffeScale(layer.name, bottom, top,
                        axis = opts.axis,
                        num_axes = opts.num_axes,
                        bias_term = opts.bias_term)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['BatchNorm']:
    opts = getopts(layer, 'batch_norm_param')
    clayer = CaffeBatchNorm(layer.name, bottom, top,
                            use_global_stats = opts.use_global_stats,
                            moving_average_fraction = opts.moving_average_fraction,
                            eps = opts.eps)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in  ['eltwise', 'Eltwise']:
    opts = getopts(layer, 'eltwise_param')
    operations = ['prod', 'sum', 'max']
    clayer = CaffeEltWise(layer.name, bottom, top,
                          operation = operations[opts.operation],
                          coeff = opts.coeff,
                          stable_prod_grad = opts.stable_prod_grad)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['data', 'Data']:
    opts = getopts(layer, 'eltwise_param')
    operations = ['prod', 'sum', 'max']
    clayer = CaffeData(layer.name, bottom, top,
                       operation = operations[opts.operation],
                       coeff = opts.coeff,
                       stable_prod_grad = opts.stable_prod_grad)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['roipooling', 'ROIPooling']:
    opts = getopts(layer, 'roi_pooling_param')
    clayer = CaffeROIPooling(layer.name, bottom, top,
                             pooled_w = opts.pooled_w,
                             pooled_h = opts.pooled_h,
                             spatial_scale = opts.spatial_scale)

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  elif ltype in ['accuracy', 'Accuracy']:
    continue

  # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  else:
    print 'Warning: unknown layer type', ltype
    continue

  if clayer is not None:
    clayer.model = cmodel
    cmodel.addLayer(clayer)
    # Fill parameters
    for dlayer in data_layers_list:
      if args.caffe_variant in ['vgg-caffe', 'caffe-old']:
        dlayer = dlayer.layer
      if dlayer.name == layer.name:
        for i, blob in enumerate(dlayer.blobs):
          blob = blobproto_to_array(blob).astype('float32')
          print '  + parameter \'%s\' <-- blob%s' % (clayer.params[i], blob.shape)
          clayer.setBlob(cmodel, i, blob)

# --------------------------------------------------------------------
#                                Get the size of the network variables
# --------------------------------------------------------------------

# Get the sizes of the network inputs
for i, inputVarName in enumerate(net.input):
  if hasattr(net, 'input_shape') and net.input_shape:
    shape = net.input_shape[i].dim._values

    # ensure that shape is a list of dimensions
    if isinstance(shape, caffe_pb2.BlobShape):
      # shape.tolist() may not preserve the order of dimensions
      shape = shape.dim._values

    shape.reverse()
  else:
    shape = [net.input_dim[k + 4*i] for k in [3,2,1,0]]

  cmodel.vars[inputVarName].shape = shape
  print '  c- Input \'{}\' is {}'.format(inputVarName, shape)

# --------------------------------------------------------------------
#                                                             Sanitize
# --------------------------------------------------------------------

# Rename layers, parametrs, and variables if they contain symbols that
# are incompatible with MatConvNet.

layerNames = cmodel.layers.keys()
for name in layerNames:
  ename = escape(name)
  if ename == name: continue
  # ensure unique
  while cmodel.layers.has_key(ename): ename = ename + 'x'
  print "Renaming layer {} to {}".format(name, ename)
  cmodel.renameLayer(name, ename)

varNames = cmodel.vars.keys()
for name in varNames:
  ename = escape(name)
  if ename == name: continue
  while cmodel.vars.has_key(ename): ename = ename + 'x'
  print "Renaming variable {} to {}".format(name, ename)
  cmodel.renameVar(name, ename)

parNames = cmodel.params.keys()
for name in parNames:
  ename = escape(name)
  if ename == name: continue
  while cmodel.params.has_key(ename): ename = ename + 'x'
  print "Renaming parameter {} to {}".format(name, ename)
  cmodel.renameParam(name, ename)

# Split in-place layers. MatConvNet handles such optimizations
# differently.

for layer in cmodel.layers.itervalues():
  if len(layer.inputs[0]) >= 1 and \
        len(layer.outputs[0]) >= 1 and \
        layer.inputs[0] == layer.outputs[0]:
    name = layer.inputs[0]
    ename = layer.inputs[0]
    while cmodel.vars.has_key(ename): ename = ename + 'x'
    print "Splitting in-place layer: renaming variable {} to {}".format(name, ename)
    cmodel.addVar(ename)
    cmodel.renameVar(name, ename, afterLayer=layer.name)
    layer.inputs[0] = name
    layer.outputs[0] = ename

# --------------------------------------------------------------------
#                                                   Get variable sizes
# --------------------------------------------------------------------

# Get the size of all other variables. This information is required
# for some special layer conversions:
#
# * For Pooling layers, fix incompatibility between padding in
#   MatConvNet and Caffe.
#
# * For Crop layers (in FCNs), determine the amount of crop (in Caffe
#   this is done at run time).

# Unflatten ROIPooling. ROIPooling will produce a H x W array instead
# of a stacked version of the same. The reshape operation below will
# convert the following InnerProduct layers in corresponding
# convolitions. This works well with transposition later.

layerNames = cmodel.layers.keys()
for name in layerNames:
  layer = cmodel.layers[name]
  if type(layer) is CaffeROIPooling:
    childrenNames = cmodel.getLayersWithInput(layer.outputs[0])
    for childName in childrenNames:
      child = cmodel.layers[childName]
      if type(child) is not CaffeInnerProduct:
        print "Error: cannot unflatten ROIPooling if this is not followed only InnerProduct layers"
        sys.exit(1)
  layer.flatten = False

cmodel.reshape()

# --------------------------------------------------------------------
#                                                                 Edit
# --------------------------------------------------------------------

# Remove dropout
if args.remove_dropout:
  layerNames = cmodel.layers.keys()
  for name in layerNames:
    layer = cmodel.layers[name]
    if type(layer) is CaffeDropout:
      print "Removing dropout layer ", name
      cmodel.renameVar(layer.outputs[0], layer.inputs[0])
      cmodel.removeLayer(name)

# Remove loss
if args.remove_loss:
  layerNames = cmodel.layers.keys()
  for name in layerNames:
    layer = cmodel.layers[name]
    if type(layer) is CaffeSoftMaxLoss:
      print "Removing loss layer ", name
      cmodel.renameVar(layer.outputs[0], layer.inputs[0])
      cmodel.removeLayer(name)

# Append softmax
for i, name in enumerate(args.append_softmax):
  # search for the layer to append SoftMax to
  if not cmodel.layers.has_key(name):
    print 'Cannot append softmax to layer {} as no such layer could be found'.format(name)
    sys.exit(1)

  if len(args.append_softmax) > 1:
    layerName = 'softmax' + (l + 1)
    outputs= ['prob' + (l + 1)]
  else:
    layerName = 'softmax'
    outputs = ['prob']

  cmodel.addLayer(CaffeSoftMax(layerName,
                               cmodel.layers[name].outputs[0:1],
                               outputs))

# Simplifications
if args.simplify:
  # Merge BatchNorm followed by Scale
  layerNames = cmodel.layers.keys()
  for name in layerNames:
    layer = cmodel.layers[name]
    if type(layer) is CaffeScale:
      if len(layer.inputs) > 1:
        continue # the scaling factor is an input, not a parameter
      if len(cmodel.getLayersWithInput(layer.inputs[0])) > 1:
        continue # other layers use the same input
      parentNames = cmodel.getLayersWithOutput(layer.inputs[0])
      if len(parentNames) != 1: continue
      parent = cmodel.layers[parentNames[0]]
      if type(parent) is not CaffeBatchNorm: continue
      smult = cmodel.params[layer.params[0]]
      sbias = cmodel.params[layer.params[1]]
      mult = cmodel.params[parent.params[0]]
      bias = cmodel.params[parent.params[1]]
      # simplification can only occur if scale layer is 1x1xC
      if smult.shape[0] != 1 or smult.shape[1] != 1: continue
      C = smult.shape[2]
      mult.value = np.reshape(smult.value, (C,)) * mult.value
      bias.value = np.reshape(smult.value, (C,)) * bias.value + \
                   np.reshape(sbias.value, (C,))
      print "Simplifying scale layer \'{}\'".format(name)
      cmodel.renameVar(layer.outputs[0], layer.inputs[0])
      cmodel.removeLayer(name)

# --------------------------------------------------------------------
#                                                        Transposition
# --------------------------------------------------------------------
#
# There are a few different conventions in MATLAB and Caffe:
#
# * In MATLAB, the frist spatial dimension is Y (vertical) followed by
#   X (horizontal), whereas in Caffe the opposite is true.
#
# * In MATLAB, images are stored in RGB format, whereas Caffe uses
#   BGR.
#
# * In MatConvNet, the first spatial coordinate is Y, whereas in Caffe
#   it is X. This affects layers such as ROI pooling.
#
# These conventions means that, if the network is directly saved in
# MCN format, then images and spatial coordinates are transposed as
# just described. While this is not a deal breaker, it is
# inconvenient.
#
# Thus we transpose all X,Y spatial dimensions in the network. For now,
# this is partially heuristic. In the future, we should add adapter layer to
# convert from MCN inputs and outputs to Caffe input and outputs and then
# simplity those away using graph transformations.

# Mark variables:
#   - requiring BGR -> RGB conversion
#   - requiring XY transposition

for i, inputVarName in enumerate(net.input):
  if inputVarName == 'data' or i == 0:
    if cmodel.vars[inputVarName].shape[2] == 3:
      cmodel.vars[inputVarName].bgrInput = (args.color_format == 'bgr')
  if not inputVarName == 'rois':
    cmodel.vars[inputVarName].transposable = True
  else:
    cmodel.vars[inputVarName].transposable = False

# Apply transformations
if args.transpose: cmodel.transpose()

cmodel.display()

# --------------------------------------------------------------------
#                                                        Normalization
# --------------------------------------------------------------------

minputs = np.empty(shape=[0,], dtype=minputdt)

# Determine the size of the inputs and input image (dataShape)
for i, inputVarName in enumerate(net.input):
  shape = cmodel.vars[inputVarName].shape
  # add metadata
  minput = np.empty(shape=[1,], dtype=minputdt)
  minput['name'][0] = inputVarName
  minput['size'][0] = row(shape)
  minputs = np.append(minputs, minput, axis=0)
  # heuristic: the first input or 'data' is the input image
  if i == 0 or inputVarName == 'data':
    dataShape = shape

print "Input image data tensor shape:", dataShape

fullImageSize = [256, 256]
if args.full_image_size:
  fullImageSize = list(make_tuple(args.full_image_size))

print "Full input image size:", fullImageSize

if average_image is not None:
  if resize_average_image:
    x = numpy.linspace(0, average_image.shape[1]-1, dataShape[0])
    y = numpy.linspace(0, average_image.shape[0]-1, dataShape[1])
    x, y = np.meshgrid(x, y, sparse=False, indexing='xy')
    average_image = bilinear_interpolate(average_image, x, y)
else:
  average_image = np.zeros((0,),dtype='float')

mnormalization = {
  'imageSize': row(dataShape),
  'averageImage': average_image,
  'interpolation': 'bilinear',
  'keepAspect': True,
  'border': row([0,0]),
  'cropSize': 1.0}

if len(fullImageSize) == 1:
  fw = max(fullImageSize[0],dataShape[1])
  fh = max(fullImageSize[0],dataShape[0])
  mnormalization['border'] = max([float(fw - dataShape[1]),
                                  float(fh - dataShape[0])])
  mnormalization['cropSize'] = min([float(dataShape[1]) / fw,
                                    float(dataShape[0]) / fh])
else:
  fw = max(fullImageSize[0],dataShape[1])
  fh = max(fullImageSize[1],dataShape[0])
  mnormalization['border'] = row([float(fw - dataShape[1]),
                                  float(fh - dataShape[0])])
  mnormalization['cropSize'] = row([float(dataShape[1]) / fw,
                                    float(dataShape[0]) / fh])

if args.caffe_variant == 'caffe_fastrcnn':
  mnormalization['interpolation'] = 'bilinear'

if args.preproc == 'caffe':
  mnormalization['interpolation'] = 'bicubic'
  mnormalization['keepAspect'] = False

print 'Input image border: ', mnormalization['border']
print 'Full input image relative crop size: ', mnormalization['cropSize']

# --------------------------------------------------------------------
#                                                              Classes
# --------------------------------------------------------------------

mclassnames = np.empty((0,), dtype=np.object)
mclassdescriptions = np.array((0,), dtype=np.object)

if synsets_wnid:
  mclassnames = np.array(synsets_wnid, dtype=np.object).reshape(1,-1)

if synsets_name:
  mclassdescriptions = np.array(synsets_name, dtype=np.object).reshape(1,-1)

mclasses = dictToMatlabStruct({'name': mclassnames,
                               'description': mclassdescriptions})

# --------------------------------------------------------------------
#                                                    Convert to MATLAB
# --------------------------------------------------------------------

# net.meta
mmeta = dictToMatlabStruct({'inputs': minputs.reshape(1,-1),
                            'normalization': mnormalization,
                            'classes': mclasses})

if args.output_format == 'dagnn':

  # This object should stay a dictionary and not a NumPy array due to
  # how NumPy saves to MATLAB

  mnet = {'layers': np.empty(shape=[0,], dtype=mlayerdt),
          'params': np.empty(shape=[0,], dtype=mparamdt),
          'meta': mmeta}

  for layer in cmodel.layers.itervalues():
    mnet['layers'] = np.append(mnet['layers'], layer.toMatlab(), axis=0)

  for param in cmodel.params.itervalues():
    mnet['params'] = np.append(mnet['params'], param.toMatlab(), axis=0)

  # to row
  mnet['layers'] = mnet['layers'].reshape(1,-1)
  mnet['params'] = mnet['params'].reshape(1,-1)

elif args.output_format == 'simplenn':

  # This object should stay a dictionary and not a NumPy array due to
  # how NumPy saves to MATLAB

  mnet = {'layers': np.empty(shape=[0,], dtype=np.object),
          'meta': mmeta}

  for layer in cmodel.layers.itervalues():
    mnet['layers'] = np.append(mnet['layers'], np.object)
    mnet['layers'][-1] = dictToMatlabStruct(layer.toMatlabSimpleNN())

  # to row
  mnet['layers'] = mnet['layers'].reshape(1,-1)

# --------------------------------------------------------------------
#                                                          Save output
# --------------------------------------------------------------------

print 'Saving network to {}'.format(args.output.name)
scipy.io.savemat(args.output, mnet, oned_as='column')
