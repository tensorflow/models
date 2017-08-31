# file: layers.py
# brief: A number of objects to wrap caffe layers for conversion
# author: Andrea Vedaldi

from collections import OrderedDict
from math import floor, ceil
from operator import mul
import numpy as np
from numpy import array
import scipy
import scipy.io
import scipy.misc
import copy
import collections

# Recent Caffes just pass a string as a type; this is used for legacy support
layers_type = {}
layers_type[0]  = 'none'
layers_type[1]  = 'accuracy'
layers_type[2]  = 'bnll'
layers_type[3]  = 'concat'
layers_type[4]  = 'conv'
layers_type[5]  = 'data'
layers_type[6]  = 'dropout'
layers_type[7]  = 'euclidean_loss'
layers_type[8]  = 'flatten'
layers_type[9]  = 'hdf5_data'
layers_type[10] = 'hdf5_output'
layers_type[28] = 'hinge_loss'
layers_type[11] = 'im2col'
layers_type[12] = 'image_data'
layers_type[13] = 'infogain_loss'
layers_type[14] = 'inner_product'
layers_type[15] = 'lrn'
layers_type[25] = 'eltwise'
layers_type[29] = 'memory_data'
layers_type[16] = 'multinomial_logistic_loss'
layers_type[17] = 'pool'
layers_type[26] = 'power'
layers_type[18] = 'relu'
layers_type[19] = 'sigmoid'
layers_type[27] = 'sigmoid_cross_entropy_loss'
layers_type[20] = 'softmax'
layers_type[21] = 'softmax_loss'
layers_type[22] = 'split'
layers_type[23] = 'tanh'
layers_type[24] = 'window_data'
layers_type[39] = 'deconvolution'
layers_type[40] = 'crop'

def getFilterOutputSize(size, kernelSize, stride, pad):
    return [floor((size[0] + pad[0]+pad[1] - kernelSize[0]) / stride[0]) + 1., \
            floor((size[1] + pad[2]+pad[3] - kernelSize[1]) / stride[1]) + 1.]

def getFilterTransform(ks, stride, pad):
    y1 = 1. - pad[0] ;
    y2 = 1. - pad[0] + ks[0] - 1 ;
    x1 = 1. - pad[2] ;
    x2 = 1. - pad[2] + ks[1] - 1 ;
    h = y2 - y1 + 1. ;
    w = x2 - x1 + 1. ;
    return CaffeTransform([h, w], stride, [(y1+y2)/2, (x1+x2)/2])

def reorder(aList, order):
    return [aList[i] for i in order]

def row(x):
    return np.array(x,dtype=float).reshape(1,-1)

def rowarray(x):
    return x.reshape(1,-1)

def rowcell(x):
    return np.array(x,dtype=object).reshape(1,-1)

def dictToMatlabStruct(d):
  if not d:
    return np.zeros((0,))
  dt = []
  for x in d.keys():
      pair = (x,object)
      if isinstance(d[x], np.ndarray): pair = (x,type(d[x]))
      dt.append(pair)
  y = np.empty((1,),dtype=dt)
  for x in d.keys():
    y[x][0] = d[x]
  return y

# --------------------------------------------------------------------
#                                                  MatConvNet in NumPy
# --------------------------------------------------------------------

mlayerdt = [('name',object),
            ('type',object),
            ('inputs',object),
            ('outputs',object),
            ('params',object),
            ('block',object)]

mparamdt = [('name',object),
            ('value',object)]

minputdt = [('name',object),
            ('size',object)]

# --------------------------------------------------------------------
#                                                      Vars and params
# --------------------------------------------------------------------

class CaffeBlob(object):
    def __init__(self, name):
        self.name = name
        self.shape = None
        self.value = np.zeros(shape=(0,0), dtype='float32')
        self.bgrInput = False
        self.transposable = True # first two dimensions are spatial

    def transpose(self):
        if self.shape: self.shape = [self.shape[k] for k in [1,0,2,3]]

    def toMatlab(self):
        mparam = np.empty(shape=[1,], dtype=mparamdt)
        mparam['name'][0] = self.name
        mparam['value'][0] = self.value
        return mparam

    def toMatlabSimpleNN(self):
        return self.value

    def hasValue(self):
        return reduce(mul, self.value.shape, 1) > 0

class CaffeTransform(object):
    def __init__(self, size, stride, offset):
        self.shape = size
        self.stride = stride
        self.offset = offset

    def __str__(self):
        return "<%s %s %s>" % (self.shape, self.stride, self.offset)

def composeTransforms(a, b):
    size = [0.,0.]
    stride = [0.,0.]
    offset = [0.,0.]
    for i in [0,1]:
        size[i] = a.stride[i] * (b.shape[i] - 1) + a.shape[i]
        stride[i] = a.stride[i] * b.stride[i]
        offset[i] = a.stride[i] * (b.offset[i] - 1) + a.offset[i]
    c = CaffeTransform(size, stride, offset)
    return c

def transposeTransform(a):
    size = [0.,0.]
    stride = [0.,0.]
    offset = [0.,0.]
    for i in [0,1]:
        size[i] = (a.shape[i] + a.stride[i] - 1.0) / a.stride[i]
        stride[i] = 1.0/a.stride[i]
        offset[i] = (1.0 + a.stride[i] - a.offset[i]) / a.stride[i]
    c = CaffeTransform(size, stride, offset)
    return c

# --------------------------------------------------------------------
#                                                               Errors
# --------------------------------------------------------------------

class ConversionError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

# --------------------------------------------------------------------
#                                                         Basic Layers
# --------------------------------------------------------------------

class CaffeLayer(object):
    def __init__(self, name, inputs, outputs):
        self.name = name
        self.inputs = inputs
        self.outputs = outputs
        self.params = []
        self.model = None

    def reshape(self, model):
        pass

    def display(self):
        print "Layer \'{}\'".format(self.name)
        print "  +- type: %s" % (self.__class__.__name__)
        print "  +- inputs: %s" % (self.inputs,)
        print "  +- outputs: %s" % (self.outputs,)
        print "  +- params: %s" % (self.params,)

    def getTransforms(self, model):
        transforms = []
        for i in enumerate(self.inputs):
            row = []
            for j in enumerate(self.outputs):
                row.append(CaffeTransform([1.,1.], [1.,1.], [1.,1.]))
            transforms.append(row)
        return transforms

    def transpose(self, model):
        pass

    def setBlob(self, model, i, blob):
        assert(False)

    def toMatlab(self):
        mlayer = np.empty(shape=[1,],dtype=mlayerdt)
        mlayer['name'][0] = self.name
        mlayer['type'][0] = None
        mlayer['inputs'][0] = rowcell(self.inputs)
        mlayer['outputs'][0] = rowcell(self.outputs)
        mlayer['params'][0] = rowcell(self.params)
        mlayer['block'][0] = dictToMatlabStruct({})
        return mlayer

    def toMatlabSimpleNN(self):
        mparam = collections.OrderedDict() ;
        mparam['name'] = self.name
        mparam['type'] = None
        return mparam

class CaffeElementWise(CaffeLayer):
    def reshape(self, model):
        for i in range(len(self.inputs)):
            model.vars[self.outputs[i]].shape = \
                model.vars[self.inputs[i]].shape

class CaffeReLU(CaffeElementWise):
    def __init__(self, name, inputs, outputs):
        super(CaffeReLU, self).__init__(name, inputs, outputs)

    def toMatlab(self):
        mlayer = super(CaffeReLU, self).toMatlab()
        mlayer['type'][0] = u'dagnn.ReLU'
        mlayer['block'][0] = dictToMatlabStruct(
            {'leak': float(0.0) })
        # todo: leak factor
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeReLU, self).toMatlabSimpleNN()
        mlayer['type'] = u'relu'
        mlayer['leak'] = float(0.0)
        return mlayer

class CaffeLRN(CaffeElementWise):
    def __init__(self, name, inputs, outputs,
                 local_size,
                 alpha,
                 beta,
                 norm_region,
                 kappa):

        super(CaffeLRN, self).__init__(name, inputs, outputs)
        self.local_size = local_size
        self.alpha = alpha
        self.beta = beta
        self.norm_region = norm_region
        self.kappa = kappa

        assert(norm_region == 'across_channels')

    def toMatlab(self):
        mlayer = super(CaffeLRN, self).toMatlab()
        mlayer['type'][0] = u'dagnn.LRN'
        mlayer['block'][0] = dictToMatlabStruct(
            {'param': row([self.local_size,
                           self.kappa,
                           self.alpha / self.local_size,
                           self.beta])})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeLRN, self).toMatlabSimpleNN()
        mlayer['type'] = u'lrn'
        mlayer['param'] = row([self.local_size,
                               self.kappa,
                               self.alpha / self.local_size,
                               self.beta])
        return mlayer

class CaffeSoftMax(CaffeElementWise):
    def __init__(self, name, inputs, outputs):
        super(CaffeSoftMax, self).__init__(name, inputs, outputs)

    def toMatlab(self):
        mlayer = super(CaffeSoftMax, self).toMatlab()
        mlayer['type'][0] = u'dagnn.SoftMax'
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeSoftMax, self).toMatlabSimpleNN()
        mlayer['type'] = u'softmax'
        return mlayer

class CaffeSoftMaxLoss(CaffeElementWise):
    def __init__(self, name, inputs, outputs):
        super(CaffeSoftMaxLoss, self).__init__(name, inputs, outputs)

    def toMatlab(self):
        mlayer = super(CaffeSoftMaxLoss, self).toMatlab()
        mlayer['type'][0] = u'dagnn.SoftMaxLoss'
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeSoftMaxLoss, self).toMatlabSimpleNN()
        mlayer['type'] = u'softmax'
        return mlayer

class CaffeDropout(CaffeElementWise):
    def __init__(self, name, inputs, outputs, ratio):
        super(CaffeDropout, self).__init__(name, inputs, outputs)
        self.ratio = ratio

    def toMatlab(self):
        mlayer = super(CaffeDropout, self).toMatlab()
        mlayer['type'][0] = u'dagnn.DropOut'
        mlayer['block'][0] = dictToMatlabStruct({'rate': float(self.ratio)})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeDropout, self).toMatlabSimpleNN()
        mlayer['type'] = u'dropout'
        mlayer['rate'] = float(self.ratio)
        return mlayer

    def display(self):
        super(CaffeDropout, self).display()
        print "  c- ratio (dropout rate):", self.ratio

class CaffeData(CaffeLayer):
    def __init__(self, name, inputs, outputs):
        super(CaffeData, self).__init__(name, inputs, outputs)

    def reshape(self, model):
        # todo: complete otehr cases
        shape = [layer.transform_param.crop_size,
                 layer.transform_param.crop_size,
                 3,
                 layer.batch_size]
        model.vars[self.outputs[0]].shape = shape

    def toMatlab(self):
        return None

    def toMatlabSimpleNN(self):
        return None

# --------------------------------------------------------------------
#                                                          Convolution
# --------------------------------------------------------------------

class CaffeConv(CaffeLayer):
    def __init__(self, name, inputs, outputs,
                 num_output,
                 bias_term,
                 pad,
                 kernel_size,
                 stride,
                 dilation,
                 group):

        super(CaffeConv, self).__init__(name, inputs, outputs)

        if len(kernel_size) == 1 : kernel_size = kernel_size * 2
        if len(stride) == 1 : stride = stride * 2
        if len(pad) == 1 : pad = pad * 4
        elif len(pad) == 2 : pad = [pad[0], pad[0], pad[1], pad[1]]

        self.num_output = num_output
        self.bias_term = bias_term
        self.pad = pad
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.group = group

        self.params = [name + '_filter']
        if bias_term: self.params.append(name + '_bias')
        self.filter_depth = None

    def display(self):
        super(CaffeConv, self).display()
        print "  +- filter dimension:", self.filter_depth
        print "  c- num_output (num filters): %s" % self.num_output
        print "  c- bias_term: %s" % self.bias_term
        print "  c- pad: %s" % (self.pad,)
        print "  c- kernel_size: %s" % self.kernel_size
        print "  c- stride: %s" % (self.stride,)
        print "  c- dilation: %s" % (self.dilation,)
        print "  c- group: %s" % (self.group,)

    def reshape(self, model):
        varin = model.vars[self.inputs[0]]
        varout = model.vars[self.outputs[0]]
        if not varin.shape: return
        varout.shape = getFilterOutputSize(varin.shape[0:2],
                                           self.kernel_size,
                                           self.stride,
                                           self.pad) \
                                           + [self.num_output, varin.shape[3]]
        self.filter_depth = varin.shape[2] / self.group

    def getTransforms(self, model):
        return [[getFilterTransform(self.kernel_size, self.stride, self.pad)]]

    def setBlob(self, model, i, blob):
        assert(i < 2)
        if i == 0:
            assert(blob.shape[0] == self.kernel_size[0])
            assert(blob.shape[1] == self.kernel_size[1])
            assert(blob.shape[3] == self.num_output)
            self.filter_depth = blob.shape[2]
        elif i == 1:
            assert(blob.shape[0] == self.num_output)
        model.params[self.params[i]].value = blob
        model.params[self.params[i]].shape = blob.shape

    def transpose(self, model):
        self.kernel_size = reorder(self.kernel_size, [1,0])
        self.stride = reorder(self.stride, [1,0])
        self.pad = reorder(self.pad, [2,3,0,1])
        self.dilation = reorder(self.dilation, [1,0])
        if model.params[self.params[0]].hasValue():
            print "Layer %s: transposing filters" % self.name
            param = model.params[self.params[0]]
            param.value = param.value.transpose([1,0,2,3])
            if model.vars[self.inputs[0]].bgrInput:
                print "Layer %s: BGR to RGB conversion" % self.name
                param.value = param.value[:,:,: : -1,:]

    def toMatlab(self):
        size = self.kernel_size + [self.filter_depth, self.num_output]
        mlayer = super(CaffeConv, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Conv'
        mlayer['block'][0] = dictToMatlabStruct(
            {'hasBias': self.bias_term,
             'size': row(size),
             'pad': row(self.pad),
             'stride': row(self.stride),
             'dilate': row(self.dilation)})
        return mlayer

    def toMatlabSimpleNN(self):
        size = self.kernel_size + [self.filter_depth, self.num_output]
        mlayer = super(CaffeConv, self).toMatlabSimpleNN()
        mlayer['type'] = u'conv'
        mlayer['weights'] = np.empty([1,len(self.params)], dtype=np.object)
        mlayer['size'] = row(size)
        mlayer['pad'] = row(self.pad)
        mlayer['stride'] = row(self.stride)
        mlayer['dilate'] = row(self.dilation)
        for p, name in enumerate(self.params):
            mlayer['weights'][0,p] = self.model.params[name].toMatlabSimpleNN()
        return mlayer

# --------------------------------------------------------------------
#                                                         InnerProduct
# --------------------------------------------------------------------

# special case: inner product
class CaffeInnerProduct(CaffeConv):
    def __init__(self, name, inputs, outputs, num_output, bias_term, axis):
        super(CaffeInnerProduct, self).__init__(name, inputs, outputs,
                                                num_output = num_output,
                                                bias_term = bias_term,
                                                pad = [0, 0, 0, 0],
                                                kernel_size = [1, 1],
                                                stride = [1, 1],
                                                dilation = [],
                                                group = 1)
        self.axis = axis
        assert(axis == 1)

    def setBlob(self, model, i, blob):
        assert(i < 1 + self.bias_term)
        if i == 0:
            self.filter_depth = blob.shape[0]
            assert(blob.shape[1] == self.num_output)
            blob = blob.reshape([1, 1, self.filter_depth, self.num_output])
        elif i == 1:
            assert(blob.shape[0] == self.num_output)
        model.params[self.params[i]].value = blob
        model.params[self.params[i]].shape = blob.shape

    def reshape(self, model):
        if not model.vars[self.inputs[0]].shape: return
        s = model.vars[self.inputs[0]].shape
        self.kernel_size = [s[0], s[1], s[2], self.num_output]
        print "Layer %s: inner product converted to filter bank of shape %s" \
            % (self.name, self.kernel_size)
        param = model.params[self.params[0]]
        if param.hasValue():
            print "Layer %s: reshaping inner product paramters of shape %s into a filter bank" % (self.name, param.value.shape)
            param.value = param.value.reshape(self.kernel_size, order='F')
        super(CaffeInnerProduct, self).reshape(model)

# --------------------------------------------------------------------
#                                                        Deconvolution
# --------------------------------------------------------------------

class CaffeDeconvolution(CaffeConv):
    def __init__(self, name, inputs, outputs,
                 num_output,
                 bias_term,
                 pad,
                 kernel_size,
                 stride,
                 dilation,
                 group):
        super(CaffeDeconvolution, self).__init__(name, inputs, outputs,
                                                 num_output = num_output,
                                                 bias_term = bias_term,
                                                 pad = pad,
                                                 kernel_size = kernel_size,
                                                 stride = stride,
                                                 dilation = dilation,
                                                 group = group)

    def setBlob(self, model, i, blob):
        assert(i < 2)
        if i == 0:
            assert(blob.shape[0] == self.kernel_size[0])
            assert(blob.shape[1] == self.kernel_size[1])
            assert(blob.shape[2] == self.num_output)
            self.filter_depth = blob.shape[3]
        elif i == 1:
            assert(blob.shape[0] == self.num_output)
        model.params[self.params[i]].value = blob
        model.params[self.params[i]].shape = blob.shape

    def reshape(self, model):
        inshape = model.vars[self.inputs[0]].shape
        if not inshape: return
        model.vars[self.outputs[0]].shape = \
            getFilterOutputSize(inshape[0:2],
                                self.kernel_size, self.stride, self.pad) + \
            [self.num_output, inshape[3]]
        self.filter_depth = inshape[2]

    def getTransforms(self, model):
        t = getFilterTransform(self.kernel_size, self.stride, self.pad)
        t = transposeTransform(t)
        return [[t]]

    def transpose(self, model):
        self.kernel_size = reorder(self.kernel_size, [1,0])
        self.stride = reorder(self.stride, [1,0])
        self.pad = reorder(self.pad, [2,3,0,1])
        if model.params[self.params[0]].hasValue():
            print "Layer %s transposing filters" % self.name
            param = model.params[self.params[0]]
            param.value = param.value.transpose([1,0,2,3])
            if model.vars[self.inputs[0]].bgrInput:
                print "Layer %s BGR to RGB conversion" % self.name
                param.value = param.value[:,:,:,: : -1]

    def toMatlab(self):
        size = self.kernel_size +  [self.num_output, self.filter_depth / self.group]
        mlayer = super(CaffeDeconvolution, self).toMatlab()
        mlayer['type'][0] = u'dagnn.ConvTranspose'
        mlayer['block'][0] = dictToMatlabStruct(
            {'hasBias': self.bias_term,
             'size': row(size),
             'upsample': row(self.stride),
             'crop': row(self.pad)})
        return mlayer

    def toMatlabSimpleNN(self):
        size = self.kernel_size + [self.num_output, self.filter_depth / self.group]
        mlayer = super(CaffeDeconvolution, self).toMatlabSimpleNN()
        mlayer['type'] = u'convt'
        mlayer['weights'] = np.empty([1,len(self.params)], dtype=np.object)
        mlayer['size'] = row(size)
        mlayer['upsample'] =  row(self.stride)
        mlayer['crop'] = row(self.pad)
        for p, name in enumerate(self.params):
            mlayer['weights'][0,p] = self.model.params[name].toMatlabSimpleNN()
        return mlayer

# --------------------------------------------------------------------
#                                                              Pooling
# --------------------------------------------------------------------

class CaffePooling(CaffeLayer):
    def __init__(self, name, inputs, outputs,
                 method,
                 pad,
                 kernel_size,
                 stride):

        super(CaffePooling, self).__init__(name, inputs, outputs)

        if len(kernel_size) == 1 : kernel_size = kernel_size * 2
        if len(stride) == 1 : stride = stride * 2
        if len(pad) == 1 : pad = pad * 4
        elif len(pad) == 2 : pad = [pad[0], pad[0], pad[1], pad[1]]

        self.method = method
        self.pad = pad
        self.kernel_size = kernel_size
        self.stride = stride

        self.pad_corrected = None

    def display(self):
        super(CaffePooling, self).display()
        print "  +- pad_corrected: %s" % (self.pad_corrected,)
        print "  c- method: ", self.method
        print "  c- pad: %s" % (self.pad,)
        print "  c- kernel_size: %s" % (self.kernel_size,)
        print "  c- stride: %s" % (self.stride,)

    def reshape(self, model):
        shape = model.vars[self.inputs[0]].shape
        if not shape: return
        # MatConvNet uses a slighly different definition of padding, which we think
        # is the correct one (it corresponds to the filters)
        self.pad_corrected = copy.deepcopy(self.pad)
        for i in [0, 1]:
            self.pad_corrected[1 + i*2] = min(
                self.pad[1 + i*2] + self.stride[i] - 1,
                self.kernel_size[i] - 1)
        model.vars[self.outputs[0]].shape = \
            getFilterOutputSize(shape[0:2],
                                self.kernel_size,
                                self.stride,
                                self.pad_corrected) + shape[2:5]

    def getTransforms(self, model):
        return [[getFilterTransform(self.kernel_size, self.stride, self.pad)]]

    def transpose(self, model):
        self.kernel_size = reorder(self.kernel_size, [1,0])
        self.stride = reorder(self.stride, [1,0])
        self.pad = reorder(self.pad, [2,3,0,1])
        if self.pad_corrected:
            self.pad_corrected = reorder(self.pad_corrected, [2,3,0,1])

    def toMatlab(self):
        mlayer = super(CaffePooling, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Pooling'
        mlayer['block'][0] = dictToMatlabStruct(
            {'method': self.method,
             'poolSize': row(self.kernel_size),
             'stride': row(self.stride),
             'pad': row(self.pad_corrected)})
        if not self.pad_corrected:
            print "Warning: pad correction for layer %s could not be computed because the layer input shape could not be determined" % (self.name)
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffePooling, self).toMatlabSimpleNN()
        mlayer['type'] = u'pool'
        mlayer['method'] = self.method
        mlayer['pool'] = row(self.kernel_size)
        mlayer['stride'] = row(self.stride)
        mlayer['pad'] = row(self.pad_corrected)
        if not self.pad_corrected:
            print "Warning: pad correction for layer %s could not be computed because the layer input shape could not be determined" % (self.name)
        return mlayer

# --------------------------------------------------------------------
#                                                           ROIPooling
# --------------------------------------------------------------------

class CaffeROIPooling(CaffeLayer):
    def __init__(self, name, inputs, outputs,
                 pooled_w,
                 pooled_h,
                 spatial_scale):
        super(CaffeROIPooling, self).__init__(name, inputs, outputs)
        self.pooled_w = pooled_w
        self.pooled_h = pooled_h
        self.spatial_scale = spatial_scale
        self.flatten = True

    def display(self):
        super(CaffeROIPooling, self).display()
        print "  c- pooled_w: %s" % (self.pooled_w,)
        print "  c- pooled_h: %s" % (self.pooled_h,)
        print "  c- spatial_scale: %s" % (self.spatial_scale,)
        print "  c- flatten: %s" % (self.flatten,)

    def reshape(self, model):
        shape1 = model.vars[self.inputs[0]].shape
        shape2 = model.vars[self.inputs[1]].shape
        if not shape1 or not shape2: return
        numChannels = shape1[2]
        numROIs = reduce(mul, shape2, 1) / 5
        if self.flatten:
            oshape =  [1,
                       1,
                       self.pooled_w * self.pooled_h * numChannels,
                       numROIs]
        else:
            oshape =  [self.pooled_w,
                       self.pooled_h,
                       numChannels,
                       numROIs]
        model.vars[self.outputs[0]].shape = oshape

    def getTransforms(self, model):
        # no transform
        return [[CaffeTransform([1.,1.], [1.,1.], [1.,1.])]]

    def transpose(self, model):
        assert(not self.flatten)
        tmp = self.pooled_w
        self.pooled_w = self.pooled_h
        self.pooled_h = tmp

    def toMatlab(self):
        mlayer = super(CaffeROIPooling, self).toMatlab()
        mlayer['type'][0] = u'dagnn.ROIPooling'
        mlayer['block'][0] = dictToMatlabStruct(
            {'subdivisions':row([self.pooled_w, self.pooled_h]),
             'transform':self.spatial_scale,
             'flatten':self.flatten})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeROIPooling, self).toMatlabSimpleNN()
        mlayer['type'] = u'roipool'
        mlayer['subdivisions'] = row([self.pooled_w, self.pooled_h])
        mlayer['transform'] = self.spatial_scale
        mlayer['flatten'] = self.flatten
        return mlayer

# --------------------------------------------------------------------
#                                                                Scale
# --------------------------------------------------------------------

class CaffeScale(CaffeLayer):
    def __init__(self, name, inputs, outputs,
                 axis,
                 num_axes,
                 bias_term):

        super(CaffeScale, self).__init__(name, inputs, outputs)

        self.axis = axis
        self.num_axes = num_axes
        self.bias_term = bias_term

        if len(self.inputs) == 1:
            self.params.append(name + '_mult')
        if len(self.inputs) < 2 and self.bias_term:
            self.params.append(name + '_bias')

        self.mult_size = [0, 0, 0, 0]

    def display(self):
        super(CaffeScale, self).display()
        print "  +- mult_size: %s" % (self.mult_size,)
        print "  c- axis: %s" % (self.axis,)
        print "  c- num_axes: %s" % (self.num_axes,)
        print "  c- bias_term: %s" % (self.bias_term,)

    def reshape(self, model):
        model.vars[self.outputs[0]].shape = model.vars[self.inputs[0]].shape

    def setBlob(self, model, i, blob):
        assert(i < self.bias_term + 1)
        # Caffe *ends* with WIDTH, we start with it, blobs are already swapped here
        k = 3 - self.axis

        # This means that the MULT dimensions are aligned to the INPUT
        # dimensions such that MULT[end] <-> INPUT[k]. For MatConvNet,
        # we simply add singletion dimensions at the beginning of MULT
        # to achieve this effect. BIAS is the same.
        mshape = tuple([1] * (k - len(blob.shape) + 1) + list(blob.shape))
        blob = blob.reshape(mshape)
        model.params[self.params[i]].value = blob
        model.params[self.params[i]].shape = blob.shape
        if i == 0: self.mult_size = blob.shape

    def getTransforms(self, model):
        # The second input can be either a variable or a paramter; in
        # both cases, there is no transform for it
        return [[CaffeTransform([1.,1.], [1.,1.], [1.,1.])]]

    def transpose(self, model):
        if len(self.inputs) == 1:
            # we only need to transpose if the scale is a parameter, not an input
            for i in range(1 + self.bias_term):
                param = model.params[self.params[i]]
                n = len(param.shape)
                if n >= 2:
                    order = range(n)
                    order[0] = 1
                    order[1] = 0
                    param.value = param.value.transpose(order)

    def toMatlab(self):
        mlayer = super(CaffeScale, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Scale'
        mlayer['block'][0] = dictToMatlabStruct(
            {'size': row(self.mult_size),
             'hasBias': self.bias_term})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeScale, self).toMatlabSimpleNN()
        # SimpleNN works only if the scaling blob is a parameter (and not a variable)
        mlayer['type'] = u'scale'
        mlayer['size'] = row(self.mult_size)
        mlayer['hasBias'] = self.bias_term
        return mlayer

# --------------------------------------------------------------------
#                                                            BatchNorm
# --------------------------------------------------------------------

class CaffeBatchNorm(CaffeLayer):
    def __init__(self, name, inputs, outputs, use_global_stats, moving_average_fraction, eps):
        super(CaffeBatchNorm, self).__init__(name, inputs, outputs)

        self.use_global_stats = use_global_stats
        self.moving_average_fraction = moving_average_fraction
        self.eps = eps

        self.params = [name + u'_mean',
                       name + u'_variance',
                       name + u'_scale_factor']

    def display(self):
        super(CaffeBatchNorm, self).display()
        print "  c- use_global_stats: %s" % (self.use_global_stats,)
        print "  c- moving_average_fraction: %s" % (self.moving_average_fraction,)
        print "  c- eps: %s" % (self.eps)

    def setBlob(self, model, i, blob):
        assert(i < 3)
        model.params[self.params[i]].value = blob
        model.params[self.params[i]].shape = blob.shape

    def reshape(self, model):
        shape = model.vars[self.inputs[0]].shape
        mean = model.params[self.params[0]].value
        variance = model.params[self.params[1]].value
        scale_factor = model.params[self.params[2]].value
        for i in range(3): del model.params[self.params[i]]
        self.params = [self.name + u'_mult',
                       self.name + u'_bias',
                       self.name + u'_moments']

        model.addParam(self.params[0])
        model.addParam(self.params[1])
        model.addParam(self.params[2])

        if shape:
            mult = np.ones((shape[2],),dtype='float32')
            bias = np.zeros((shape[2],),dtype='float32')
            model.params[self.params[0]].value = mult
            model.params[self.params[0]].shape = mult.shape
            model.params[self.params[1]].value = bias
            model.params[self.params[1]].shape = bias.shape

        if mean.size:
            moments = np.concatenate(
                (mean.reshape(-1,1) / scale_factor,
                 np.sqrt(variance.reshape(-1,1) / scale_factor + self.eps)),
                axis=1)
            model.params[self.params[2]].value = moments
            model.params[self.params[2]].shape = moments.shape

        model.vars[self.outputs[0]].shape = shape

    def toMatlab(self):
        mlayer = super(CaffeBatchNorm, self).toMatlab()
        mlayer['type'][0] = u'dagnn.BatchNorm'
        mlayer['block'][0] = dictToMatlabStruct(
            {'epsilon': self.eps})
        return mlayer

    def toMatlabSimpleNN(self):
        mlayer = super(CaffeBatchNorm, self).toMatlabSimpleNN()
        mlayer['type'] = u'bnorm'
        mlayer['epsilon'] = self.eps
        return mlayer

# --------------------------------------------------------------------
#                                                               Concat
# --------------------------------------------------------------------

class CaffeConcat(CaffeLayer):
    def __init__(self, name, inputs, outputs, concatDim):
        super(CaffeConcat, self).__init__(name, inputs, outputs)
        self.concatDim = concatDim

    def transpose(self, model):
        self.concatDim = [1, 0, 2, 3][self.concatDim]

    def reshape(self, model):
        sizes = [model.vars[x].shape for x in self.inputs]
        osize = copy.deepcopy(sizes[0])
        osize[self.concatDim] = 0
        for thisSize in sizes:
            for i in range(len(thisSize)):
                if self.concatDim == i:
                    osize[i] = osize[i] + thisSize[i]
                else:
                    if osize[i] != thisSize[i]:
                        print "Warning: concat layer: inconsistent input dimensions", sizes
        model.vars[self.outputs[0]].shape = osize

    def display(self):
        super(CaffeConcat, self).display()
        print "  Concat Dim: ", self.concatDim

    def toMatlab(self):
        mlayer = super(CaffeConcat, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Concat'
        mlayer['block'][0] = dictToMatlabStruct({'dim': float(self.concatDim) + 1})
        return mlayer

    def toMatlabSimpleNN(self):
        raise ConversionError('Concat layers do not work in a SimpleNN network')

# --------------------------------------------------------------------
#                                                   EltWise (Sum, ...)
# --------------------------------------------------------------------

class CaffeEltWise(CaffeElementWise):
    def __init__(self, name, inputs, outputs,
                 operation,
                 coeff,
                 stable_prod_grad):
        super(CaffeEltWise, self).__init__(name, inputs, outputs)
        self.operation = operation
        self.coeff = coeff
        self.stable_prod_grad = stable_prod_grad

    def toMatlab(self):
        mlayer = super(CaffeEltWise, self).toMatlab()
        if self.operation == 'sum':
            mlayer['type'][0] = u'dagnn.Sum'
        else:
            # not implemented
            assert(False)
        return mlayer

    def display(self):
        super(CaffeEltWise, self).display()
        print "  c- operation: ", self.operation
        print "  c- coeff: %s" % self.coeff
        print "  c- stable_prod_grad: %s" % self.stable_prod_grad

    def reshape(self, model):
        model.vars[self.outputs[0]].shape = \
            model.vars[self.inputs[0]].shape
        for i in range(1, len(self.inputs)):
            assert(model.vars[self.inputs[0]].shape == model.vars[self.inputs[i]].shape)

    def toMatlabSimpleNN(self):
        raise ConversionError('EltWise (sum, ...) layers do not work in a SimpleNN network')

# --------------------------------------------------------------------
#                                                                 Crop
# --------------------------------------------------------------------

class CaffeCrop(CaffeLayer):
    def __init__(self, name, inputs, outputs):
        super(CaffeCrop, self).__init__(name, inputs, outputs)
        self.crop = []

    def display(self):
        super(CaffeCrop, self).display()
        print "  Crop: %s" % self.crop

    def reshape(self, model):
        # this is quite complex as we need to compute on the fly
        # the geometry
        tfs1 = model.getParentTransforms(self.inputs[0], self.name)
        tfs2 = model.getParentTransforms(self.inputs[1], self.name)

        print
        print self.name, self.inputs[0]
        for a,x in enumerate(tfs1): print "%10s %s" % (x,tfs1[x])
        print self.name, self.inputs[1]
        for a,x in enumerate(tfs2): print "%10s %s" % (x,tfs2[x])

        # the goal is to crop inputs[0] to make it as big as inputs[1] and
        # aligned to it; so now we find the map from inputs[0] to inputs[1]

        tf = None
        for name, tf2 in tfs2.items():
            if tfs1.has_key(name):
                tf1 = tfs1[name]
                tf = composeTransforms(transposeTransform(tf2), tf1)
                break
        if tf is None:
            print "Error: could not find common ancestor for inputs '%s' and '%s' of the CaffeCrop layer '%s'" % (self.inputs[0], self.inputs[1], self.name)
            sys.exit(1)
        print "  Transformation %s -> %s = %s" % (self.inputs[0],
                                                  self.inputs[1], tf)
        # for this to make sense it shoudl be tf.stride = 1
        assert(tf.stride[0] == 1 and tf.stride[1] == 1)

        # finally we can get the crops!
        self.crop = [0.,0.]
        for i in [0,1]:
            # i' = alpha (i - 1) + beta + crop = 1 for i = 1
            # crop = 1 - beta
            self.crop[i] =  round(1 - tf.offset[i])
        print "  Crop %s" % self.crop

        # print
        # print "resolved"
        # tfs3 = model.getParentTransforms(self.outputs[0])
        # for a,x in enumerate(tfs3): print "%10s %s" % (x,tfs3[x])

        # now compute output variable size, which will be the size of the second input
        model.vars[self.outputs[0]].shape = model.vars[self.inputs[1]].shape

    def getTransforms(self, model):
        t = CaffeTransform([1.,1.], [1.,1.], [1.+self.crop[0],1.+self.crop[1]])
        return [[t],[None]]

    def toMatlab(self):
        mlayer = super(CaffeCrop, self).toMatlab()
        mlayer['type'][0] = u'dagnn.Crop'
        mlayer['block'][0] = dictToMatlabStruct({'crop': row(self.crop)})
        return mlayer

    def toMatlabSimpleNN(self):
        # todo: simple 1 input crop layers should be supported though!
        raise ConversionError('Crop layers do not work in a SimpleNN network')

# --------------------------------------------------------------------
#                                                          Caffe Model
# --------------------------------------------------------------------

class CaffeModel(object):
    def __init__(self):
        self.layers = OrderedDict()
        self.vars = OrderedDict()
        self.params = OrderedDict()

    def addLayer(self, layer):
        ename = layer.name
        while self.layers.has_key(ename):
            ename = ename + 'x'
        if layer.name != ename:
            print "Warning: a layer with name %s was already found, using %s instead" % \
                (layer.name, ename)
            layer.name = ename
        for v in layer.inputs:  self.addVar(v)
        for v in layer.outputs: self.addVar(v)
        for p in layer.params: self.addParam(p)
        self.layers[layer.name] = layer

    def addVar(self, name):
        if not self.vars.has_key(name):
            self.vars[name] = CaffeBlob(name)

    def addParam(self, name):
        if not self.params.has_key(name):
            self.params[name] = CaffeBlob(name)

    def renameLayer(self, old, new):
        self.layers[old].name = new
        # reinsert layer with new name -- this mess is to preserve the order
        layers = OrderedDict([(new,v) if k==old else (k,v)
                              for k,v in self.layers.items()])
        self.layers = layers

    def renameVar(self, old, new, afterLayer=None):
        self.vars[old].name = new
        if afterLayer is not None:
            start = self.layers.keys().index(afterLayer) + 1
        else:
            start = 0
        # fix all references to the variable
        for layer in self.layers.values()[start:-1]:
            layer.inputs = [new if x==old else x for x in layer.inputs]
            layer.outputs = [new if x==old else x for x in layer.outputs]
        self.vars[new] = copy.deepcopy(self.vars[old])
        # check if we can delete the old one (for afterLayet != None)
        stillUsed = False
        for layer in self.layers.values():
            stillUsed = stillUsed or old in layer.inputs or old in layer.outputs
        if not stillUsed:
            del self.vars[old]

    def renameParam(self, old, new):
        self.params[old].name = new
        # fix all references to the variable
        for layer in self.layers.itervalues():
            layer.params = [new if x==old else x for x in layer.params]
        var = self.params[old]
        del self.params[old]
        self.params[new] = var

    def removeParam(self, name):
        del self.params[name]

    def removeLayer(self, name):
        # todo: fix this stuff for weight sharing
        layer = self.layers[name]
        for paramName in layer.params:
            self.removeParam(paramName)
        del self.layers[name]

    def getLayersWithOutput(self, varName):
        layerNames = []
        for layer in self.layers.itervalues():
            if varName in layer.outputs:
                layerNames.append(layer.name)
        return layerNames

    def getLayersWithInput(self, varName):
        layerNames = []
        for layer in self.layers.itervalues():
            if varName in layer.inputs:
                layerNames.append(layer.name)
        return layerNames

    def reshape(self):
        for layer in self.layers.itervalues():
            layer.reshape(self)

    def display(self):
        for layer in self.layers.itervalues():
            layer.display()
        for var in self.vars.itervalues():
            print 'Variable \'{}\''.format(var.name)
            print '   + shape (computed): %s' % (var.shape,)
        for par in self.params.itervalues():
            print 'Parameter \'{}\''.format(par.name)
            print '   + data found: %s' % (par.shape is not None)
            print '   + data shape: %s' % (par.shape,)

    def transpose(self):
        for var in self.vars.itervalues():
            if var.transposable: var.transpose()
        for layer in self.layers.itervalues():
            layer.transpose(self)

    def getParentTransforms(self, variableName, topLayerName=None):
        layerNames = self.layers.keys()
        if topLayerName:
            layerIndex = layerNames.index(topLayerName)
        else:
            layerIndex = len(self.layers) + 1
        transforms = OrderedDict()
        transforms[variableName] = CaffeTransform([1.,1.], [1.,1.], [1.,1.])
        for layerName in reversed(layerNames[0:layerIndex]):
            layer = self.layers[layerName]
            layerTfs = layer.getTransforms(self)
            for i, inputName in enumerate(layer.inputs):
                tfs = []
                if transforms.has_key(inputName):
                    tfs.append(transforms[inputName])
                for j, outputName in enumerate(layer.outputs):
                    if layerTfs[i][j] is None: continue
                    if transforms.has_key(outputName):
                        composed = composeTransforms(layerTfs[i][j], transforms[outputName])
                        tfs.append(composed)

                if len(tfs) > 0:
                    # should resolve conflicts, not simply pick the first tf
                    transforms[inputName] = tfs[0]
        return transforms
