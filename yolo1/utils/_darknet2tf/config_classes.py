"""
This file contains the layers (Config objects) that are used by the Darknet
config file parser.

For more details on the layer types and layer parameters, visit https://github.com/AlexeyAB/darknet/wiki/CFG-Parameters-in-the-different-layers

Currently, the parser is incomplete and we can only guarantee that it works for
models in the YOLO family (YOLOv3 and older).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from typing import Tuple, Sequence, List


class Config(ABC):
    """
    The base class for all layers that are used by the parser. Each subclass
    defines a new layer type. Most nodes correspond to distinct layers that
    appear in the final network. [net] corresponds to the input to the model.

    Each subclass must be a @dataclass and must have the following fields:
    ```{python}
        _type: str = None
        w: int = field(init=True, repr=True, default=0)
        h: int = field(init=True, repr=True, default=0)
        c: int = field(init=True, repr=True, default=0)
    ```

    These fields are used when linking different layers together, but weren't
    included in the Config class due to limitations in the dataclasses package.
    (w, h, c) will correspond to the different input dimensions of a DarkNet
    layer: the width, height, and number of channels.
    """
    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        '''
        Output shape of the layer. The output must be a 3-tuple of ints
        corresponding to the the width, height, and number of channels of the
        output.

        Returns:
            A tuple corresponding to the output shape of the layer.
        '''
        return

    def load_weights(self, files) -> int:
        '''
        Load the weights for the current layer from a file.

        Arguments:
            files: Open IO object for the DarkNet weights file

        Returns:
            the number of bytes read.
        '''
        return 0

    def get_weights(self) -> list:
        '''
        Returns:
            a list of Numpy arrays consisting of all of the weights that
            were loaded from the weights file
        '''
        return []

    @classmethod
    def from_dict(clz, net, layer_dict) -> "Config":
        '''
        Create a layer instance from the previous layer and a dictionary
        containing all of the parameters for the DarkNet layer. This is how
        linking is done by the parser.
        '''
        if 'w' not in layer_dict:
            prevlayer = net[-1]
            l = {
                "w": prevlayer.shape[0],
                "h": prevlayer.shape[1],
                "c": prevlayer.shape[2],
                **layer_dict
            }
        else:
            l = layer_dict
        return clz(**l)

    @abstractmethod
    def to_tf(self, tensors):
        """
        Convert the DarkNet configuration object to a tensor given the previous
        tensors that occoured in the network. This function should also return
        a Keras layer if it has weights.

        Returns:
            if weights: a tuple consisting of the output tensor and Keras layer
            if no weights: the output tensor
        """
        return None


class _LayerBuilder(dict):
    """
    This class defines a registry for the layer builder in the DarkNet weight
    parser. It allows for syntactic sugar when registering Config subclasses to
    the parser.
    """
    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError as e:
            raise KeyError(f"Unknown layer type: {key}") from e

    def register(self, *layer_types: str):
        '''
        Register a parser node (layer) class with the layer builder.
        '''
        def decorator(clz):
            for layer_type in layer_types:
                self[layer_type] = clz
            return clz

        return decorator


layer_builder = _LayerBuilder()


@layer_builder.register('conv', 'convolutional')
@dataclass
class convCFG(Config):
    _type: str = None
    w: int = field(init=True, repr=True, default=0)
    h: int = field(init=True, repr=True, default=0)
    c: int = field(init=True, repr=True, default=0)

    size: int = field(init=True, repr=True, default=0)
    stride: int = field(init=True, repr=True, default=0)
    pad: int = field(init=True, repr=True, default=0)
    filters: int = field(init=True, repr=True, default=0)
    activation: str = field(init=True, repr=False, default='linear')
    groups: int = field(init=True, repr=False, default=1)
    batch_normalize: int = field(init=True, repr=False, default=0)
    dilation: int = field(init=True, repr=False, default=1)

    nweights: int = field(repr=False, default=0)
    biases: np.array = field(repr=False, default=None)  #
    weights: np.array = field(repr=False, default=None)
    scales: np.array = field(repr=False, default=None)
    rolling_mean: np.array = field(repr=False, default=None)
    rolling_variance: np.array = field(repr=False, default=None)

    def __post_init__(self):
        self.pad = (self.size - 1) // 2
        self.nweights = int(
            (self.c / self.groups) * self.filters * self.size * self.size)
        return

    @property
    def shape(self):
        w = len_width(self.w, self.size, self.pad, self.stride)
        h = len_width(self.h, self.size, self.pad, self.stride)
        return (w, h, self.filters)

    def load_weights(self, files):
        self.biases = read_n_floats(self.filters, files)
        bytes_read = self.filters

        if self.batch_normalize == 1:
            self.scales = read_n_floats(self.filters, files)
            self.rolling_mean = read_n_floats(self.filters, files)
            self.rolling_variance = read_n_floats(self.filters, files)
            bytes_read += self.filters * 3

        # used as a guide:
        # https://github.com/thtrieu/darkflow/blob/master/darkflow/dark/convolution.py
        weights = read_n_floats(self.nweights, files)
        self.weights = weights.reshape(self.filters, self.c, self.size,
                                       self.size).transpose([2, 3, 1, 0])
        bytes_read += self.nweights
        return bytes_read * 4

    def get_weights(self, printing=False):
        if printing:
            print(
                "[weights, biases, biases, scales, rolling_mean, rolling_variance]"
            )
        if self.batch_normalize:
            return [
                self.weights,
                self.scales,  #gamma
                self.biases,  #beta
                self.rolling_mean,
                self.rolling_variance
            ]
        else:
            return [self.weights, self.biases]

    def to_tf(self, tensors):
        from official.vision.beta.projects.yolo.modeling.layers.nn_blocks import DarkConv
        layer = DarkConv(
            filters=self.filters,
            kernel_size=(self.size, self.size),
            strides=(self.stride, self.stride),
            padding='same',
            dilation_rate=(self.dilation, self.dilation),
            use_bn=bool(self.batch_normalize),
            activation=activation_function_dn_to_keras_name(self.activation),
        )  # TODO: Where does groups go
        return layer(tensors[-1]), layer

@layer_builder.register('local')
@dataclass
class localCfg(Config):
    # implementation based on:
    # https://github.com/thtrieu/darkflow/blob/master/darkflow/dark/convolution.py
    # l.4-l.25
    _type: str = None
    w: int = field(init=True, repr=True, default=0)
    h: int = field(init=True, repr=True, default=0)
    c: int = field(init=True, repr=True, default=0)

    size: int = field(init=True, repr=True, default=0)
    stride: int = field(init=True, repr=True, default=0)
    pad: int = field(init=True, repr=True, default=0)
    filters: int = field(init=True, repr=True, default=0)
    activation: str = field(init=True, repr=False, default='linear')
    groups: int = field(init=True, repr=False, default=1)

    nweights: int = field(repr=False, default=0)
    weights: np.array = field(repr=False, default=None)
    biases: np.array = field(repr=False, default=None)

    def __post_init__(self):
        self.pad = int(self.pad) * int(self.size / 2) if self.size != 1 else 0

        w = len_width(self.w, self.size, self.pad, self.stride)
        h = len_width(self.h, self.size, self.pad, self.stride)
        self.nweights = int(
            self.filters * self.size * self.size)
        return

    @property
    def shape(self):
        w = len_width(self.w, self.size, self.pad, self.stride)
        h = len_width(self.h, self.size, self.pad, self.stride)
        return (w, h, self.filters)

    def load_weights(self, files):
        w = len_width(self.w, self.size, self.pad, self.stride)
        h = len_width(self.h, self.size, self.pad, self.stride)
        self.biases = read_n_floats(w * h * self.filters, files)

        bytes_read = self.filters * w * h

        weights = read_n_floats(self.nweights, files)
        # self.weights = weights.reshape(self.h * self.w, self.filters, self.c, self.size,
        #                                self.size).transpose([0, 3, 4, 2, 1])
        bytes_read += self.nweights
        return bytes_read * 4

    def get_weights(self, printing=False):
        if printing:
            print(
                "[weights, biases]"
            )
        return [self.weights, self.biases]

    def to_tf(self, tensors):
        from tensorflow.keras.layers import LocallyConnected2D, ZeroPadding2D, LeakyReLU

        zero_pad_layer = ZeroPadding2D(
            padding=self.pad
        )

        if self.activation == "leaky":
            self.activation = LeakyReLU(alpha=0.1)

        local_layer = LocallyConnected2D(
            filters=self.filters,
            kernel_size=(self.size, self.size),
            strides=(self.stride, self.stride),
            padding='valid',  # currently LocallyConnected2D only supports 'valid
            activation=activation_function_dn_to_keras_name(self.activation),
        )

        return local_layer(zero_pad_layer(tensors[-1])), local_layer
        #return local_layer(tensors[-1]), local_layer

@layer_builder.register('shortcut')
@dataclass
class shortcutCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    _from: List[int] = field(init=True, default_factory=list)
    activation: str = field(init=True, default='linear')

    @property
    def shape(self):
        return (self.w, self.h, self.c)

    @classmethod
    def from_dict(clz, net, layer_dict):
        '''
        Create a layer instance from the previous layer and a dictionary
        containing all of the parameters for the DarkNet layer. This is how
        linking is done by the parser.
        '''
        _from = layer_dict['from']
        if type(_from) is not tuple:
            _from = (_from, )

        prevlayer = net[-1]
        l = {
            "_type": layer_dict['_type'],
            "w": prevlayer.shape[0],
            "h": prevlayer.shape[1],
            "c": prevlayer.shape[2],
            "_from": _from,
            "activation": layer_dict['activation'],
        }
        return clz(**l)

    def to_tf(self, tensors):
        from tensorflow.keras.layers import add
        from tensorflow.keras.activations import get
        activation = get(activation_function_dn_to_keras_name(self.activation))

        my_tensors = [tensors[-1]]
        for i in self._from:
            my_tensors.append(tensors[i])

        return activation(add(my_tensors))


@layer_builder.register('route')
@dataclass
class routeCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    layers: List[int] = field(init=True, default_factory=list)
    groups: int = field(repr=False, default=1)
    group_id: int = field(repr=False, default=0)

    @property
    def shape(self):
        return (self.w, self.h, self.c // self.groups)

    @classmethod
    def from_dict(clz, net, layer_dict):
        # Calculate shape of the route
        layers = layer_dict['layers']
        if type(layers) is tuple:
            layers_iter = iter(layers)
            w, h, c = net[next(layers_iter)].shape
            for l in layers_iter:
                lw, lh, lc = net[l].shape
                if (lw, lh) != (w, h):
                    raise ValueError(
                        f"Width and heights of route layer [#{len(net)}] inputs {layers} do not match.\n   Previous: {(w, h)}\n   New: {(lw, lh)}"
                    )
                c += lc
        else:
            w, h, c = net[layers].shape
            layers = (layers, )
        assert c % layer_dict.get(
            'groups', 1
        ) == 0, "The number of channels must evenly divide among the groups."

        # Create layer
        l = layer_dict.copy()
        l["w"] = w
        l["h"] = h
        l["c"] = c
        l["layers"] = layers
        return clz(**l)

    def to_tf(self, tensors):
        import tensorflow as tf
        from tensorflow.keras.layers import concatenate

        if len(self.layers) == 1:
            stacked = tensors[self.layers[0]]
        else:
            my_tensors = []
            for i in self.layers:
                my_tensors.append(tensors[i])
            stacked = concatenate(my_tensors)

        if self.groups == 1:
            return stacked
        else:
            return tf.split(stacked, self.groups, axis=-1)[self.group_id]


@layer_builder.register('net', 'network')
@dataclass
class netCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    @property
    def shape(self):
        return (self.w, self.h, self.c)

    @classmethod
    def from_dict(clz, net, layer_dict):
        assert len(
            net.data
        ) == 0, "A [net] section cannot occour in the middle of a DarkNet model"
        l = {
            "_type": layer_dict["_type"],
            "w": layer_dict["width"],
            "h": layer_dict["height"],
            "c": layer_dict["channels"]
        }
        return clz(**l)

    def to_tf(self, tensors):
        from tensorflow.keras import Input
        return Input(shape=[self.w, self.h, self.c])


@layer_builder.register('yolo')
@dataclass
class yoloCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    mask: List[int] = field(init=True, default_factory=list)
    anchors: List[Tuple[int, int]] = field(init=True, default_factory=list)
    scale_x_y: int = field(init=True, default=1)

    @property
    def shape(self):
        return (self.w, self.h, self.c)

    @classmethod
    def from_dict(clz, net, layer_dict):
        prevlayer = net[-1]
        l = {
            "_type": layer_dict['_type'],
            "mask": layer_dict['mask'],
            "anchors": layer_dict['anchors'],
            "w": prevlayer.shape[0],
            "h": prevlayer.shape[1],
            "c": prevlayer.shape[2]
        }
        return clz(**l)

    def to_tf(self, tensors):
        return tensors[-1]  # TODO: Fill out


@layer_builder.register('upsample')
@dataclass
class upsampleCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    stride: int = field(init=True, default=2)

    @property
    def shape(self):
        return (self.stride * self.w, self.stride * self.h, self.c)

    def to_tf(self, tensors):
        from tensorflow.keras.layers import UpSampling2D
        return UpSampling2D(size=(self.stride, self.stride))(tensors[-1])


@layer_builder.register('maxpool')
@dataclass
class maxpoolCFG(Config):
    _type: str = None
    w: int = field(init=True, default=0)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    stride: int = field(init=True, default=2)
    size: int = field(init=True, default=2)

    @property
    def shape(self):
        pad = 0 if self.stride == 1 else 1
        #print((self.w//self.stride, self.h//self.stride, self.c))
        return (
            self.w // self.stride, self.h // self.stride, self.c
        )  #((self.w - self.size) // self.stride + 2, (self.h - self.size) // self.stride + 2, self.c)

    def to_tf(self, tensors):
        #from tensorflow.nn import max_pool2d
        from tensorflow.keras.layers import MaxPooling2D
        return MaxPooling2D(pool_size=(self.size, self.size),
                            strides=(self.stride, self.stride),
                            padding='same')(tensors[-1])


@layer_builder.register('connected')
@dataclass
class connectedCFG(Config):
    # Used as guide: https://github.com/thtrieu/darkflow/blob/master/darkflow/dark/connected.py
    _type: str = None
    w: int = field(init=True, repr=True, default=0)
    h: int = field(init=True, repr=True, default=0)
    c: int = field(init=True, repr=True, default=0)

    output: int = field(init=True, repr=True, default=1715)
    activation: str = field(init=True, repr=False, default='linear')

    nweights: int = field(repr=False, default=0)
    biases: np.array = field(repr=False, default=None)
    weights: np.array = field(repr=False, default=None)

    def __post_init__(self):
        # number of weights (exlucding bias = input size * output size)
        self.nweights = int(
            self.c * self.w * self.h * self.output)
        return

    @property
    def shape(self):
        return (self.output,)

    def load_weights(self, files):
        self.biases = read_n_floats(self.output, files)
        bytes_read = self.output
        weights = read_n_floats(self.nweights, files)
        #self.weights = weights.reshape(self.c * self.w * self.h, self.output)
        bytes_read += self.nweights
        return bytes_read * 4

    def get_weights(self, printing=False):
        if printing:
            print("[weights, biases]")
        return [self.weights, self.biases]

    def to_tf(self, tensors):
        from tensorflow.keras.layers import Dense, Flatten

        layer1 = Flatten()
        layer2 = Dense(
            self.output,
            activation=activation_function_dn_to_keras_name(self.activation)
        )
        return layer2(layer1(tensors[-1])), layer2

@layer_builder.register('detection')
@dataclass
class detectionCFG(Config):
    _type: str = None
    w: int = field(init=True, repr=True, default=0)
    h: int = field(init=True, repr=True, default=0)
    c: int = field(init=True, repr=True, default=0)

    classes: int = field(init=True, repr=True, default=20)
    coords: int = field(init=True, repr=True, default=4)
    rescore: int = field(init=True, repr=True, default=1)
    side: int = field(init=True, repr=True, default=7)
    num: int = field(init=True, repr=True, default=3)
    softmax: int = field(init=True, repr=True, default=0)
    sqrt: int = field(init=True, repr=True, default=1)
    jitter: float = field(init=True, repr=True, default=0.2)

    object_scale: int = field(init=True, repr=True, default=1)
    noobject_scale: int = field(init=True, repr=True, default=0.5)
    class_scale: int = field(init=True, repr=True, default=1)
    coord_scale: int = field(init=True, repr=True, default=5)

    @classmethod
    def from_dict(clz, net, layer_dict):
        if 'w' not in layer_dict:
            prevlayer = net[-1]
            l = {
                "w": prevlayer.shape[0],
                **layer_dict
            }
        else:
            l = layer_dict
        return clz(**l)

    @property
    def shape(self):
        return (self.side, self.side, self.num * 5 + self.classes)

    def to_tf(self, tensors):
        from tensorflow.keras.layers import Reshape
        shape = (self.side, self.side, self.num * 5 + self.classes)
        layer = Reshape(shape)
        return layer(tensors[-1])


@layer_builder.register('dropout')
@dataclass
class dropoutCFG(Config):
    _type: str = None
    w: int = field(init=True, default=1715)
    h: int = field(init=True, default=0)
    c: int = field(init=True, default=0)

    probability: int = field(init=True, default=0.5)

    @property
    def shape(self):
        return (self.w, self.h, self.c)

    def to_tf(self, tensors):
        from tensorflow.keras.layers import Dropout
        dropout = Dropout(rate=self.probability)
        return dropout(tensors[-1])


def len_width(n, f, p, s):
    '''
    n: height or width
    f: kernels height or width
    p: padding
    s: strides height or width
    '''
    return int(((n + 2 * p - f) / s) + 1)


def len_width_up(n, f, p, s):
    '''
    n: height or width
    f: kernels height or width
    p: padding
    s: strides height or width
    '''
    return int(((n - 1) * s - 2 * p + (f - 1)) + 1)


def read_n_floats(n, bfile):
    """c style read n float 32"""
    return np.fromfile(bfile, 'f4', n)


def read_n_int(n, bfile, unsigned=False):
    """c style read n int 32"""
    dtype = '<u4' if unsigned else '<i4'
    return np.fromfile(bfile, dtype, n)


def read_n_long(n, bfile, unsigned=False):
    """c style read n int 64"""
    dtype = '<u8' if unsigned else '<i8'
    return np.fromfile(bfile, dtype, n)


def activation_function_dn_to_keras_name(dn):
    return dn


def get_primitive_tf_layer_name(var, piece=3):
    name = var.name
    parts = name.rsplit('/', piece)
    if len(parts) < piece:
        return None

    token = parts[-piece]
    cid = []
    while True:
        try:
            name, count = token.rsplit('_', 1)
        except:
            break
        try:
            cid.append(int(count))
        except:
            break
        else:
            token = name

    if token[0].upper() == token[0]:
        return cid, token
    return cid, ''.join([x.capitalize() for x in token.split('_')])


# Testing locally connected config:
if __name__ == "__main__":
    config_path = "yolo/utils/_darknet2tf/test_locally_connected_config.cfg"
    weights_path = "D:/yolov1.weights"

    from yolo.utils import DarkNetConverter
    converter = DarkNetConverter()
    x = converter.read(config_file=config_path, weights_file=weights_path)
    print("Weights loaded successfully")
    x = x.to_tf()
    print("Layers converted to TF successfully")
