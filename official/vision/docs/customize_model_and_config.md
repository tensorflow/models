# Customize Config and Model


## Overview

The TF Vision library contains a collection of state-of-the-art models for a
variety of tasks, including image classification, object detection, and
segmentation. It is usually a good idea to start with an existing model when
developing a new one. This is because there is often a lot of work that has
already been done to develop and test the existing model, so you can save time
and money by reusing it and you can customize them to meet your specific needs.

The existing model can be found
[here](https://github.com/tensorflow/models/tree/master/official/vision/modeling).
They are well-tested and have been shown to work well on a variety of tasks.
Also,
[TF-Vision Model ZOO](https://github.com/tensorflow/models/blob/master/official/vision/MODEL_GARDEN.md)
allows you to browse the available models, read documentation, and download
models.

## Customize Model

### Build your model with lego blocks

A TFMG model is composed by stacking pre-built/tested reusable modules including
e.g. backbones, decoders, and headers. For example, an object detection model
can be viewed as a stack of

```python
    Input data --> Backbone --> Decoders --> Header --> Output
```

Therefore, having a customized model by customizing your choice of different
reusable modules is a good starting point.

The
[backbone](https://github.com/tensorflow/models/tree/master/official/vision/modeling/backbones)
is the foundational part of the model responsible for feature extraction. It
typically consists of layers that process the input data and progressively
extracts features of increasing complexity. Popular choices for backbones
include architectures like
[ResNet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/resnet.py),
[MobileNet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py),
and
[EfficientNet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/efficientnet.py).

The
[decoder](https://github.com/tensorflow/models/tree/master/official/vision/modeling/decoders)
modules typically follow the backbone and are responsible for transforming the
extracted features into task-specific representations. The decoders can
effectively process/convert/combine the features from the backbone into desired
shapes/resolutions that are suitable for the task. Popular choices for decoders
include architectures like
[FPN](https://github.com/tensorflow/models/blob/master/official/vision/modeling/decoders/fpn.py),
[NASFPN](https://github.com/tensorflow/models/blob/master/official/vision/modeling/decoders/nasfpn.py),
and
[ASPP](https://github.com/tensorflow/models/blob/master/official/vision/modeling/decoders/aspp.py).

The
[header](https://github.com/tensorflow/models/blob/master/official/vision/modeling/heads)
is usually the final part of the model and is responsible for making predictions
based on the features obtained from the decoders or backbones directly. The
header's architecture depends on the specific task at hand. For instance, in
object detection, the
[RPNhead](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/heads/dense_prediction_heads.py#L497)
is used to generate a large number of proposals, which are then passed to the
second stage detector, and
[MaskHead](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/heads/instance_heads.py#L222)
is used after the RPN head to predict a mask for each proposal which is used to
refine the bounding box and improve the segmentation of the object.

The TensorFlow Model Garden's structure allows for customization at different
levels. You can experiment by combining the above different backbone
architectures, Decoder, and Header or modify existing ones to better suit your
specific task requirements.

#### Example

Customization becomes achievable by selecting a pre-existing model and then
varying the combinations of available backbones, decoders, or headers.

1.  To choose a model, you can browse the Model Garden's list of
    [models](https://github.com/tensorflow/models/tree/master/official/vision/modeling).
    Each model represents a specific task that it can be used for. For instance,
    image classification experiment uses
    [Classification Model](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/classification_model.py#L25)
    and Object Detection experiment uses
    [Retinanet Model](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/retinanet_model.py#L26).

2.  To Configure other components, depending on your requirements you can select
    the
    [backbones](https://github.com/tensorflow/models/tree/master/official/vision/modeling/backbones),
    [decoders](https://github.com/tensorflow/models/tree/master/official/vision/modeling/decoders),
    and
    [headers](https://github.com/tensorflow/models/tree/master/official/vision/modeling/heads)
    from their respective folders. Combine the above chosen components to create
    your tailored model configuration.

Refer to a
[Semantic Segmentation](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/configs/semantic_segmentation.py#L686)
experiment `mnv2_deeplabv3_cityscapes`. This approach combines the efficiency of
`MobileNetV2` with the semantic segmentation capabilities of `DeepLabV3` to
create a powerful tool for segmenting urban scenes in the `Cityscapes` dataset.
It combines the backbone
[mobilenet](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/mobilenet.py)
, decoder
[ASPP](https://github.com/tensorflow/models/blob/master/official/vision/modeling/decoders/aspp.py),
and header
[Segmentation_heads](https://github.com/tensorflow/models/blob/master/official/vision/modeling/heads/segmentation_heads.py)
along with the existing
[Segmentation Model](https://github.com/tensorflow/models/blob/master/official/vision/modeling/segmentation_model.py).

### Creating Customized Model with Existing modules

Custom models can be useful in a variety of situations, such as, if there is a
specific and unique problem to solve that isn't addressed by existing models,
users may need to create a custom model to address that issue.

#### Instructions

To create a custom model , user need to follow the below steps:

*   **Create a subclass Class and define model architecture**

    To customize a model in TensorFlow, users can define the model architecture
    by subclassing the `tf.keras.Model` which allows us to define our own custom
    layers, methods and parameters. If you subclass `tf.keras.Model`, you can
    define the architecture in the `__init__` function and the forward pass
    computation in the `call` function.

    However we make a distinction based on the scenario. In simpler cases, such
    as a single input tensor ,e.g.,
    [Classification](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/classification_model.py#L97),
    we typically opt for a `functional-subclass` style for simplicity. In this
    style, users only need to override the `__init__` function and not the
    `call` function. For more intricate situations like Detection or Instance
    segmentation, we employ the `subclass` style.

*   **Implement Methods in the Subclass**

    It is recommended that descendant of `tf.keras.Model` implement the
    following methods:

    *   `__init__()` : This method is used to construct the modules that make up
        the model. By subclassing the `tf.keras.Model` class, you should define
        your layers in the `__init__()` method.

    *   `call()`: Implement the call method, which defines the forward pass of
        your model.This is where you apply any custom operations or
        modifications specific to your task. Also note that when opting for a
        `functional-subclass` style, there's no necessity to override the
        `call()` method.

    *   `checkpoint_items()`: This method is used to define the checkpoint
        strategy for the model. It returns a dictionary of items to be
        additionally checkpointed.

    *   `get_config()` : Config is a serializable python dictionary containing
        the configuration of the model.You can use this method to return a
        python dictionary containing the model's configuration.

    *   `from_config()` : This method is called when the model is deserialized
        from a configuration. You can use this method to create a new instance
        of the model from its configuration.

    *   Also, adding additional `@property` is a suggested approach for
        conveniently accessing essential attributes such as the backbone,
        decoder, and header.

    Here is an example of how to create a custom model in TensorFlow using
    subclassing:

    ```python
    class customModel(tf.keras.Model):
        def __init__(self, backbone: tf.keras.Model,
                           decoder: tf.keras.Model,
                           head: tf.keras.layers.Layer,
                           num_classes: int,
                           input_specs: tf.keras.layers.InputSpec =
                                           layers.InputSpec(shape=………),………,**kwargs):

              super(customModel, self).__init__(**kwargs)
              self._config_dict = {
                  'backbone': backbone,
                  'decoder': decoder,
                  'head': head,
                    ………
               }
              self.backbone = backbone
              self.decoder = decoder
              self.head = head
              ………

        def call(self, inputs: tf.Tensor, training: bool = None
          ) -> Dict[str, tf.Tensor]:
            backbone_features = self.backbone(inputs)
            decoder_features = self.decoder(backbone_features)
            ………

            logits = self.head((backbone_features, decoder_features))
            outputs = {'logits': logits}
        return outputs

        @property
        def checkpoint_items(
          self)->Mapping[str,Union[tf.keras.Model,tf.keras.layers.Layer]

        items = dict(
          backbone=self.backbone,
          head=self.head)
        if self.decoder is not None:
          items.update(decoder=self.decoder)
        return items

        @property
        def backbone(self) -> tf.keras.Model:
          return self._backbone

        @property
        def decoder(self) -> tf.keras.Model:
          return self._decoder

        @property
        def head(self) -> tf.keras.layers.Layer:
          return self._head

        def get_config(self)-> Mapping[str, Any]:
          return self._config_dict

        @classmethod
        def from_config(cls, config):
          return cls(**config)
    ```

    <br>

    The arguments passed to the `__init__` method are primarily InputSpec,
    backbone, decoder, and head. But user can freely add as many arguments as
    needed.

    **InputSpec** - `tf.keras.layers.InputSpec` is a class that is used to
    specify the shape and data type of input tensors for a network layer. It is
    typically used in the `__init__` method of a custom layer to specify the
    expected shape and data type of the input tensor. Above is the example of
    how to use InputSpec in a custom layer.

    **backbone, decoder, and head :** Users can create model by assembling
    individual components, such as backbones, decoders, header. This modular
    approach allows for better organization, reusability, and flexibility when
    building complex models.

<br>

*   **Build factory method to construct custom model**

    Users can define a function that takes a model config as input and returns a
    `customModel` instance, similar to the example `build_customModel` function
    below. This function is the main entry point to build a model usually
    present in the
    [factory class](https://github.com/tensorflow/models/blob/master/official/vision/modeling/factory.py).
    An
    [example](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/resnet.py)
    of building a classification model is ResNet.

    ```python
    def build_customModel(input_specs:tf.keras.layers.InputSpec,
         model_config: example_cfg.ExampleModel,
         ………
         backbone: Optional[tf.keras.Model] = None,
         decoder: Optional[tf.keras.Model] = None
          **kwargs) -> tf.keras.Model:
      ………

       if not backbone:
          backbone = backbones.factory.build_backbone(………)

       if not decoder:
          decoder = decoders.factory.build_decoder(………)
        head = model_heads.(………)
      ………

    return customModel(
     num_classes=model_config.num_classes, backbone, decoder,
     num_classes=model_config.num_classes, backbone, decoder,
    ```

    <br>

*   **Build model in Task Class**

    A task is a subclass of
    [base_task.Task](https://github.com/tensorflow/models/blob/master/official/core/base_task.py)
    that defines model, input, loss, metric and one training and evaluation
    step, etc. Tasks class provides artifacts for training/validation
    procedures, including loading/iterating over Datasets, training/validation
    steps, calculating the loss and customized metrics with reduction.

    ```python
    class ExampleTask(base_task.Task):

        def build_model(self) -> tf.keras.Model:

            input_specs = tf.keras.layers.InputSpec(shape=[None] +
                      self.task_config.model.input_size)

            model = factory.build_customModel(
                     input_specs=input_specs,
                     model_config=self.task_config.model,
                     ………
            ………
            return model
    ```

#### Example

Here is an example of how to implement a Segmentation model using individual
components. This experiment `seg_deeplabv3_pascal` uses the `dilated_resnet`
[backbone](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/configs/semantic_segmentation.py#L240),
`aspp`
[decoder](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/configs/semantic_segmentation.py#L238)
and `SegmentationHead`
[header](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/configs/semantic_segmentation.py#L252).

                                              |     |
--------------------------------------------- | ---
Segmentation models class                     | [segmentation_model.py](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/segmentation_model.py)
Factory methods to build models               | [build_segmentation_model](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/factory.py#L378)
Segmentation task definition                  | [semantic_segmentation.py](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/tasks/semantic_segmentation.py#L35)
Image classification configuration definition | [semantic_segmentation.py](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/configs/semantic_segmentation.py#L131)

### Creating Customized Backbones, Decoders, or Headers

In the TensorFlow Model Garden, a typical high-level structure of a model
includes three main components: backbone, decoders, and header. This modular
structure allows for customization at different levels.

#### Customize Backbones

The backbone processes the input data and produces a feature map, which contains
high-level representations of the input.

Creating customized backbones in the TensorFlow Model Garden involves designing
and implementing your own feature extraction networks tailored to your specific
needs. Here's a general outline of how you might approach this process:

*   **Define the Backbone Class:** Create a new Python class that defines your
    customized backbone architecture. This class should inherit from
    TensorFlow's `tf.keras.Model` class.

*   **Build the Architecture:** Within your backbone class, define the layers
    and connections that make up your backbone architecture. You might also
    consider using building blocks provided by TFM, such as
    [residual blocks](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/layers/nn_blocks.py#L57),
    [depthwise separable convolutions block](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/modeling/layers/nn_blocks.py#L2181)
    or other options.

*   **Implement Methods in the Subclass:** It is recommended that descendant of
    `tf.keras.Model` implement the following methods:

    *   `__init__()` : This method is called when the backbone is first created.
        By subclassing the `tf.keras.Model` class, you should define your layers
        in the `__init__()` method. You can use this method to initialize the
        backbone's weights and parameters.

    *   `get_config()` : Config is a serializable python dictionary containing
        the configuration of the backbone.You can use this method to return a
        python dictionary containing the backbone's configuration.

    *   `from_config()` : This method is called when the backbone is
        deserialized from a configuration. You can use this method to create a
        new instance of the backbone from its configuration.

    In addition to these methods, you may also need to override other methods,
    such as `summary()` and `save_weights()`, depending on your specific needs.

*   **Define a backbone builder and annotated by factory method for
    registration :** One can register a new backbone model by importing the
    factory and register the build in the backbone file. For Example,
    `@factory.register_backbone_builder('custom_backbone')` supports
    registration of `custom_backbone` class.

    Here's a sample of what the code structure might look like:

    ```python
    class custom_backbone(tf.keras.Model):
      def __init__(self,

                model_id: str,
                input_specs: tf.keras.layers.InputSpec =
                                      layers.InputSpec(shape=[None, None, None, 3]),
                kernel_initializer: str = 'VarianceScaling',
                kernel_regularizer: tf.keras.regularizers.Regularizer = None,
                bias_regularizer: tf.keras.regularizers.Regularizer = None,
                activation: str = 'relu',
                se_inner_activation: str = 'relu',
                norm_momentum: float = 0.99,………,**kwargs):

                self._model_id = model_id
                self._input_specs = input_specs
                self._se_ratio = se_ratio
                ………
                # Build intermediate blocks.
                inputs =tf.keras.Input(shape=input_specs.shape[1:])
                          x = layers.Conv2D(
                filters=int(64 * stem_depth_multiplier),
                kernel_size=7,
                strides=2,
                use_bias=False,
                padding='same',
                kernel_initializer=self._kernel_initializer,
                kernel_regularizer=self._kernel_regularizer,
                bias_regularizer=self._bias_regularizer,
                )(inputs)
                x = self._norm(
                ………
                x = layers.MaxPool2D(pool_size=3, strides=2,padding='same')(x)
                return x

      def get_config(self):
          config_dict = {

              'model_id': self._model_id,
              'activation': self._activation,
              'norm_momentum': self._norm_momentum,
              'kernel_initializer': self._kernel_initializer,
              'kernel_regularizer': self._kernel_regularizer,
              'bias_regularizer': self._bias_regularizer}
        return config_dict

      @classmethod
      def from_config(cls, config, custom_objects=None):
        return cls(**config)

    @factory.register_backbone_builder('custom_backbone')
    def build_custom_backbone(
        input_specs: tf.keras.layers.InputSpec,
        backbone_config: hyperparams.Config,
        norm_activation_config: hyperparams.Config,
        l2_regularizer: tf.keras.regularizers.Regularizer = None)
                                              ->  tf.keras.Model:
        """Builds backbone from a config."""
        backbone_type = backbone_config.type
        backbone_cfg = backbone_config.get()
        assert backbone_type == 'custom_backbone',(f'Inconsistent
                                            backbone type '
                                            f'{backbone_type}')
        return custom_backbone(
            model_id=backbone_cfg.model_id,
            input_specs=input_specs,
            ………
            activation=norm_activation_config.activation,
            norm_momentum=norm_activation_config.norm_momentum,
            kernel_regularizer=l2_regularizer,
            bn_trainable=backbone_cfg.bn_trainable)
    ```

    <br>

*   **Add to the *init* file to make it accessible:** Import the custom backbone
    class and add a build in **init**.py. Add it to this
    [file](https://github.com/tensorflow/models/blob/master/official/vision/modeling/backbones/__init__.py).

*   **Add a dedicated config** : Create a separate config specifically for your
    custom backbone in the backbones
    [configurations file](https://github.com/tensorflow/models/blob/master/official/vision/configs/backbones.py).

    ```python
    @dataclasses.dataclass
    class Custom_backbone(hyperparams.Config):

       model_id: str = '100'
       stochastic_depth_drop_rate: float = 0.0
       se_ratio: float = 0.0
       ………
    ```

    <br>

*   **Add an entry to the ensemble `Backbone` config class:** Include a new
    entry to the configuration class `class Backbone(hyperparams.OneOfConfig)`of
    the Backbone.

#### Customize Decoder

The
[decoders](https://github.com/tensorflow/models/blob/master/official/vision/modeling/decoders)
are responsible for taking the feature map produced by the backbone and
generating predictions for specific tasks.

The customization of the `Decoder` procedure closely resembles the
`Customize Backbones` The steps involved in this process might involve adjusting parameters or architecture to meet specific requirements or preferences.

#### Customize Header

The
[header](https://github.com/tensorflow/models/tree/master/official/vision/modeling/heads)
is the final component of the model and is responsible for producing the final
output. It takes the refined features from the decoders and applies additional
operations to generate the desired output.

The customization of the `Header` procedure closely resembles the
`Customize Backbones`. The steps involved in this process might involve adjusting parameters or
architecture to meet specific requirements or preferences.

By separating the model into these three components, TensorFlow Model Garden
allows for easy customization at different levels. Users can choose different
pre-trained backbones or even design their own backbone architecture. They can
also customize the decoders to suit their specific task requirements.
Additionally, the header can be modified to adapt the model to different output
formats or to add additional layers for fine-tuning or transfer learning. This
modular structure enables flexibility and allows users to build and customize
models for a wide range of vision tasks using TensorFlow Model Garden.

## Customize Config

Customizing the configuration allows you to experiment with different
hyperparameters and architectures for your model and allows you to tailor the
behavior of your model and the training process to better suit your specific
task and requirements. By defining a separate Config class, you can easily
adjust the values of different hyperparameters and other configuration details
without modifying the model architecture itself. This approach can also make it
easier to compare the performance of different configurations and tune your
model more effectively.

### Instructions

To create a custom configuration for your experiment , user need to follow the
below steps:

*   **Customize Module Configs (as needed)** :

    **Input config :** Create `class CustomDataConfig(cfg.DataConfig)`.The
    CustomDataConfig class should subclass
    [DataConfig](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/core/config_definitions.py#L28),
    the base configuration for building datasets. It contains the configurations
    related to input data.The parameters of the config class may vary, as it
    depends on what fields and configuration settings you want to customize for
    your data, you can add more fields as needed. Here is an example of what an
    Input config class might contain:

    ```python
    @dataclasses.dataclass
    class CustomDataConfig(cfg.DataConfig):

      input_path: str = ''
      global_batch_size: int = 0
      is_training: bool = True
      dtype: str = 'float32'
      shuffle_buffer_size: int = 10000
      cycle_length: int = 10
      file_type: str = 'tfrecord'
      ………
    ```

    **The model config :** Create `class CustomModel(hyperparams.Config)`.The
    CustomModel class should subclass
    [hyperparams.Config](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/modeling/hyperparams/base_config.py#L66),
    the base configuration class that supports YAML/JSON based overrides. This
    class is used to declare custom model parameters. Here's an example code
    snippet:

    ```python
    @dataclasses.dataclass
    class CustomModel(hyperparams.Config):

      num_classes: int = 0
      input_size: List[int]= dataclasses.field(default_factory=list)
      backbone: ………
      dropout_rate: float = 0.0
      ………
    ```

    **Loss and Evaluation config :** Create `class Losses(hyperparams.Config)`
    for loss related configuration and `class Evaluation(hyperparams.Config)`
    for evaluation metrics configuration.The `Losses` and `Evaluation` class
    should subclass
    [hyperparams.Config](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/modeling/hyperparams/base_config.py#L66).
    Refer below example code snippet:

    ```python
    @dataclasses.dataclass
    class Losses(hyperparams.Config):

      l2_weight_decay: float = 0.0
      loss_weight: float = 1.0
      one_hot: bool = True
      label_smoothing: float = 0.0
      ………
    ```

    ```python
    @dataclasses.dataclass
    class Evaluation(hyperparams.Config):

       top_k: int = 5
       precision_and_recall_thresholds: Optional[List[float]] = None
       report_per_class_precision_and_recall: bool = False
       ………
    ```

    **The task config :** Create `class CustomTask(cfg.TaskConfig)`.The
    CustomTask class should subclass
    [TaskConfig](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/core/config_definitions.py#L289).
    It contains the configurations passed to task class. It consolidates all the
    above i.e input config, model config, loss and evaluation config; and can be
    passed within an experiment easily as an object.

    Here is an example of what a task config class might contain:

    ```python
    @dataclasses.dataclass
    class CustomTask(cfg.TaskConfig):
        model: CustomModel = CustomModel()
        train_data: CustomDataConfig =
                    CustomDataConfig(is_training=True)
        validation_data: CustomDataConfig =
                    CustomDataConfig(is_training=False)
        losses: Losses = Losses()
        evaluation: Evaluation = Evaluation()
        freeze_backbone: bool = False
        ………
    ```

    All the above configs are defined as dataclass objects, for storing data
    objects.

*   **Define Experiment**

    To create an experiment, the user can define a method, infuse it with
    default parameters and generate the
    [ExperimentConfig](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/core/config_definitions.py#L308)
    object as output. Use
    [tfm.core.exp_factory.register_config_factory](https://www.tensorflow.org/api_docs/python/tfm/core/exp_factory/register_config_factory)
    to register ExperimentConfig factory method with a unique name. Users can
    create as many experiments as needed.

    [ExperimentConfig](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/core/config_definitions.py#L308)
    contains the configurations passed to the corresponding experiment. It
    consolidates
    [TaskConfig](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/core/config_definitions.py#L289)
    objects discussed previously,
    [TrainerConfig](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/core/config_definitions.py#L211)
    and
    [RuntimeConfig](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/core/config_definitions.py#L140)
    objects.

    Refer below example experiment with runtime, task and trainer config and
    `example_experiment` as an unique name to the experiment.

    ```python
    @exp_factory.register_config_factory('example_experiment')

    def vision_example_experiment() -> cfg.ExperimentConfig:
        train_batch_size = 4096
        eval_batch_size = 4096
        steps_per_epoch = 10

    config = cfg.ExperimentConfig(
        runtime=cfg.RuntimeConfig(enable_xla=True),
        task=CustomTask(
            model=CustomModel(
                num_classes=1001,
                input_size=[224, 224, 3],
                backbone=………,

            losses=Losses(l2_weight_decay=1e-4),
            train_data=CustomDataConfig(input_path=………),
            validation_data=CustomDataConfig(input_path=………),
        trainer=cfg.TrainerConfig(
            steps_per_loop=steps_per_epoch,
            summary_interval=steps_per_epoch,
            ………
            optimizer_config=optimization.OptimizationConfig({
                'optimizer': {
                    'type': 'sgd',
                      ………
                },
                'learning_rate': {
                    'type': 'stepwise',
                        ………
                },
                'warmup': {
                    'type': 'linear',
                      ………
                }
            })),
          )
      ………
    return config
    ```

*   **Create YAML file** Finally, create a YAML file to override default
    parameter values of the above experiment. By storing all relevant
    hyperparameters and settings in a YAML file, you can more easily track and
    manage changes to your experiment configurations. This can make it easier to
    reproduce or modify experiments.

    ```yaml
    runtime:
      distribution_strategy: 'tpu'
      mixed_precision_dtype: 'bfloat16'
    task:
      model:
        num_classes: 1001
        input_size: [128, 128, 3]
      train_data:
        input_path: ………
        ………
      validation_data:
        input_path: ………
        ………
    trainer:
        steps_per_loop: 312
        summary_interval: 312
        ………
        optimizer_config:
           optimizer:
             type: 'sgd'
             ………
        learning_rate:
             type: 'stepwise'
             ………
    ```

### Example

Refer to the example of
[Image classification](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/configs/image_classification.py#L29)
configuration
definition, [experiments](https://github.com/tensorflow/models/blob/7f239d8ec19b5c2d44e0d5aa2a09dbea0da6d737/official/vision/configs/image_classification.py#L122)
and
[YAML](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/image_classification/imagenet_mobilenetv1_tpu.yaml)
file.
