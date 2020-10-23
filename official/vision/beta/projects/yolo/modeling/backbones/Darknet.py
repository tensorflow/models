import tensorflow as tf
import tensorflow.keras as ks
import collections

from official.vision.beta.modeling.backbones import factory
from official.vision.beta.projects.yolo.modeling import building_blocks as nn_blocks


# builder required classes
class BlockConfig(object):

  def __init__(self, layer, stack, reps, bottleneck, filters, kernel_size,
               strides, padding, activation, route, output_name, is_output):
    '''
        get layer config to make code more readable

        Args:
            layer: string layer name
            reps: integer for the number of times to repeat block
            filters: integer for the filter for this layer, or the output depth
            kernel_size: integer or none, if none, it implies that the the building block handles this automatically. not a layer input
            downsample: boolean, to down sample the input width and height
            output: boolean, true if the layer is required as an output
        '''
    self.layer = layer
    self.stack = stack
    self.repetitions = reps
    self.bottleneck = bottleneck
    self.filters = filters
    self.kernel_size = kernel_size
    self.strides = strides
    self.padding = padding
    self.activation = activation
    self.route = route
    self.output_name = output_name
    self.is_output = is_output
    return


def build_block_specs(config):
  specs = []
  for layer in config:
    specs.append(BlockConfig(*layer))
  return specs


def darkconv_config_todict(config, kwargs):
  dictvals = {
      "filters": config.filters,
      "kernel_size": config.kernel_size,
      "strides": config.strides,
      "padding": config.padding
  }
  dictvals.update(kwargs)
  return dictvals


def darktiny_config_todict(config, kwargs):
  dictvals = {"filters": config.filters, "strides": config.strides}
  dictvals.update(kwargs)
  return dictvals


def maxpool_config_todict(config, kwargs):
  return {
      "pool_size": config.kernel_size,
      "strides": config.strides,
      "padding": config.padding,
      "name": kwargs["name"]
  }


class layer_registry(object):

  def __init__(self):
    self._layer_dict = {
        "DarkTiny": (nn_blocks.DarkTiny, darktiny_config_todict),
        "DarkConv": (nn_blocks.DarkConv, darkconv_config_todict),
        "MaxPool": (tf.keras.layers.MaxPool2D, maxpool_config_todict)
    }
    return

  def _get_layer(self, key):
    return self._layer_dict[key]

  def __call__(self, config, kwargs):
    layer, get_param_dict = self._get_layer(config.layer)
    param_dict = get_param_dict(config, kwargs)
    return layer(**param_dict)


# model configs
LISTNAMES = [
    "default_layer_name", "level_type", "number_of_layers_in_level",
    "bottleneck", "filters", "kernal_size", "strides", "padding",
    "default_activation", "route", "level/name", "is_output"
]

CSPDARKNET53 = {
    "list_names": LISTNAMES,
    "splits": {"backbone_split": 106,
               "neck_split": 138},
    "backbone": [
        ["DarkConv", None, 1, False, 32, 3, 1, "same", "mish", -1, 0, False],  # 1
        ["DarkRes", "csp", 1, True, 64, None, None, None, "mish", -1, 1, False],  # 3
        ["DarkRes", "csp", 2, False, 128, None, None, None, "mish", -1, 2, False],  # 2
        ["DarkRes", "csp", 8, False, 256, None, None, None, "mish", -1, 3, True],
        ["DarkRes", "csp", 8, False, 512, None, None, None, "mish", -1, 4, True],  # 3
        ["DarkRes", "csp", 4, False, 1024, None, None, None, "mish", -1, 5, True],  # 6  #route
    ]
}

DARKNET53 = {
    "list_names": LISTNAMES,
    "splits": {"backbone_split": 76},
    "backbone": [
        ["DarkConv", None, 1, False, 32, 3, 1, "same", "leaky", -1, 0, False],  # 1
        ["DarkRes", "residual", 1, True, 64, None, None, None, "leaky", -1, 1, False],  # 3
        ["DarkRes", "residual", 2, False, 128, None, None, None, "leaky", -1, 2, False],  # 2
        ["DarkRes", "residual", 8, False, 256, None, None, None, "leaky", -1, 3, True],
        ["DarkRes", "residual", 8, False, 512, None, None, None, "leaky", -1, 4, True],  # 3
        ["DarkRes", "residual", 4, False, 1024, None, None, None, "leaky", -1, 5, True],  # 6
    ]
}

CSPDARKNETTINY = {
    "list_names": LISTNAMES,
    "splits": {"backbone_split": 28},
    "backbone": [
        ["DarkConv", None, 1, False, 32, 3, 2, "same", "leaky", -1, 0, False],  # 1
        ["DarkConv", None, 1, False, 64, 3, 2, "same", "leaky", -1, 1, False],  # 1
        ["CSPTiny", "csp_tiny", 1, False, 64, 3, 2, "same", "leaky", -1, 2, False],  # 3
        ["CSPTiny", "csp_tiny", 1, False, 128, 3, 2, "same", "leaky", -1, 3, False],  # 3
        ["CSPTiny", "csp_tiny", 1, False, 256, 3, 2, "same", "leaky", -1, 4, True],  # 3
        ["DarkConv", None, 1, False, 512, 3, 1, "same", "leaky", -1, 5, True],  # 1
    ]
}

DARKNETTINY = {
    "list_names": LISTNAMES,
    "splits": {"backbone_split": 14},
    "backbone": [
        ["DarkConv", None, 1, False, 16, 3, 1, "same", "leaky", -1, 0, False],  # 1
        ["DarkTiny", None, 1, True, 32, 3, 2, "same", "leaky", -1, 1, False],  # 3
        ["DarkTiny", None, 1, True, 64, 3, 2, "same", "leaky", -1, 2, False],  # 3
        ["DarkTiny", None, 1, False, 128, 3, 2, "same", "leaky", -1, 3, False],  # 2
        ["DarkTiny", None, 1, False, 256, 3, 2, "same", "leaky", -1, 4, True],
        ["DarkTiny", None, 1, False, 512, 3, 2, "same", "leaky", -1, 5, False],  # 3
        ["DarkTiny", None, 1, False, 1024, 3, 1, "same", "leaky", -1, 5, True],  # 6  #route
    ]
}

BACKBONES = {
    "darknettiny": DARKNETTINY,
    "darknet53": DARKNET53,
    "cspdarknet53": CSPDARKNET53,
    "cspdarknettiny": CSPDARKNETTINY
}


@ks.utils.register_keras_serializable(package='yolo')
class Darknet(ks.Model):

  def __init__(
      self,
      model_id="darknet53",
      input_shape=tf.keras.layers.InputSpec(shape=[None, None, None, 3]),
      min_size=None,
      max_size=5,
      activation=None,
      use_sync_bn=False,
      norm_momentum=0.99,
      norm_epsilon=0.001,
      kernel_initializer='glorot_uniform',
      kernel_regularizer=None,
      bias_regularizer=None,
      config=None,
      **kwargs):

    layer_specs, splits = Darknet.get_model_config(model_id)

    self._model_name = model_id
    self._splits = splits
    self._input_shape = input_shape
    self._registry = layer_registry()

    # default layer look up
    self._min_size = min_size
    self._max_size = max_size
    self._output_specs = None

    self._kernel_initializer = kernel_initializer
    self._bias_regularizer = bias_regularizer
    self._norm_momentum = norm_momentum
    self._norm_epislon = norm_epsilon
    self._use_sync_bn = use_sync_bn
    self._activation = activation
    self._weight_decay = kernel_regularizer

    self._default_dict = {
        "kernel_initializer": self._kernel_initializer,
        "weight_decay": self._weight_decay,
        "bias_regularizer": self._bias_regularizer,
        "norm_momentum": self._norm_momentum,
        "norm_epsilon": self._norm_epislon,
        "use_sync_bn": self._use_sync_bn,
        "activation": self._activation,
        "name": None
    }

    inputs = ks.layers.Input(shape=self._input_shape.shape[1:])
    output = self._build_struct(layer_specs, inputs)
    super().__init__(inputs=inputs, outputs=output, name=self._model_name)
    return

  @property
  def input_specs(self):
    return self._input_shape

  @property
  def output_specs(self):
    return self._output_specs

  @property
  def splits(self):
    return self._splits

  def _build_struct(self, net, inputs):
    endpoints = collections.OrderedDict()
    stack_outputs = [inputs]
    for i, config in enumerate(net):
      if config.stack == None:
        x = self._build_block(stack_outputs[config.route],
                              config,
                              name=f"{config.layer}_{i}")
        stack_outputs.append(x)
      elif config.stack == "residual":
        x = self._residual_stack(stack_outputs[config.route],
                                 config,
                                 name=f"{config.layer}_{i}")
        stack_outputs.append(x)
      elif config.stack == "csp":
        x = self._csp_stack(stack_outputs[config.route],
                            config,
                            name=f"{config.layer}_{i}")
        stack_outputs.append(x)
      elif config.stack == "csp_tiny":
        x_pass, x = self._tiny_stack(stack_outputs[config.route],
                                     config,
                                     name=f"{config.layer}_{i}")
        stack_outputs.append(x_pass)
      if (config.is_output and
          self._min_size == None):
        endpoints[config.output_name] = x
      elif self._min_size != None and config.output_name >= self._min_size and config.output_name <= self._max_size:
        endpoints[config.output_name] = x

    self._output_specs = {l: endpoints[l].get_shape() for l in endpoints.keys()}
    return endpoints

  def _get_activation(self, activation):
    if self._activation == None:
      return activation
    else:
      return self._activation

  def _csp_stack(self, inputs, config, name):
    if config.bottleneck:
      csp_filter_reduce = 1
      residual_filter_reduce = 2
      scale_filters = 1
    else:
      csp_filter_reduce = 2
      residual_filter_reduce = 1
      scale_filters = 2
    self._default_dict["activation"] = self._get_activation(config.activation)
    self._default_dict["name"] = f"{name}_csp_down"
    x, x_route = nn_blocks.CSPDownSample(filters=config.filters,
                                         filter_reduce=csp_filter_reduce,
                                         **self._default_dict)(inputs)
    for i in range(config.repetitions):
      self._default_dict["name"] = f"{name}_{i}"
      x = nn_blocks.DarkResidual(filters=config.filters // scale_filters,
                                 filter_scale=residual_filter_reduce,
                                 **self._default_dict)(x)

    self._default_dict["name"] = f"{name}_csp_connect"
    output = nn_blocks.CSPConnect(filters=config.filters,
                                  filter_reduce=csp_filter_reduce,
                                  **self._default_dict)([x, x_route])
    self._default_dict["activation"] = self._activation
    self._default_dict["name"] = None
    return output

  def _tiny_stack(self, inputs, config, name):
    self._default_dict["activation"] = self._get_activation(config.activation)
    self._default_dict["name"] = f"{name}_tiny"
    x, x_route = nn_blocks.CSPTiny(filters=config.filters,
                                   **self._default_dict)(inputs)
    self._default_dict["activation"] = self._activation
    self._default_dict["name"] = None
    return x, x_route

  def _residual_stack(self, inputs, config, name):
    self._default_dict["activation"] = self._get_activation(config.activation)
    self._default_dict["name"] = f"{name}_residual_down"
    x = nn_blocks.DarkResidual(filters=config.filters,
                               downsample=True,
                               **self._default_dict)(inputs)
    for i in range(config.repetitions - 1):
      self._default_dict["name"] = f"{name}_{i}"
      x = nn_blocks.DarkResidual(filters=config.filters,
                                 **self._default_dict)(x)
    self._default_dict["activation"] = self._activation
    self._default_dict["name"] = None
    return x

  def _build_block(self, inputs, config, name):
    x = inputs
    i = 0
    self._default_dict["activation"] = self._get_activation(config.activation)
    while i < config.repetitions:
      self._default_dict["name"] = f"{name}_{i}"
      layer = self._registry(config, self._default_dict)
      x = layer(x)
      i += 1
    self._default_dict["activation"] = self._activation
    self._default_dict["name"] = None
    return x

  @staticmethod
  def get_model_config(name):
    name = name.lower()
    backbone = BACKBONES[name]["backbone"]
    splits = BACKBONES[name]["splits"]
    return build_block_specs(backbone), splits


@factory.register_backbone_builder('darknet')
def build_darknet(
    input_specs: tf.keras.layers.InputSpec,
    model_config,
    l2_regularizer: tf.keras.regularizers.Regularizer = None) -> tf.keras.Model:

  backbone_type = model_config.backbone.type
  backbone_cfg = model_config.backbone.get()
  norm_activation_config = model_config.norm_activation
  model = Darknet(model_id=backbone_cfg.model_id,
                 input_shape=input_specs,
                 activation=norm_activation_config.activation,
                 use_sync_bn=norm_activation_config.use_sync_bn,
                 norm_momentum=norm_activation_config.norm_momentum,
                 norm_epsilon=norm_activation_config.norm_epsilon,
                 kernel_regularizer=l2_regularizer)
  model.summary()
  return model
