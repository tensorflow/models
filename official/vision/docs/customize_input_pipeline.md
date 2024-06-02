# Customize Input Pipeline






## Overview


A task is a class that encapsulates the logic of loading data, building models,
performing one-step training and validation, etc. It connects all components
together and is called by the base
[Trainer](https://github.com/tensorflow/models/blob/master/official/core/base_trainer.py).
You can create your own task by inheriting from base
[Task](https://github.com/tensorflow/models/blob/master/official/core/base_task.py),
or from one of the
[tasks](https://github.com/tensorflow/models/tree/master/official/vision/tasks)
we already defined, if most of the operations can be reused. An `ExampleTask`
inheriting from
[ImageClassificationTask](https://github.com/tensorflow/models/blob/master/official/vision/tasks/image_classification.py#L31)
can be found
[here](https://github.com/tensorflow/models/blob/master/official/vision/examples/starter/example_task.py).


In a task class, the `build_inputs` method is responsible for building the input
pipeline for training and evaluation. Specifically, it will instantiate a
Decoder object and a Parser object, which are used to create an `InputReader`
that will generate a `tf.data.Dataset` object.


Here's an example code snippet that demonstrates how to create a custom
`build_inputs` method:


```python
def build_inputs(
    self,
    params: exp_cfg.DataConfig,
    input_context: Optional[tf.distribute.InputContext] = None
) -> tf.data.Dataset:
  ....


  decoder = sample_input.Decoder()
  parser = sample_input.Parser(
      output_size=..., num_classes=...)
  reader = input_reader_factory.input_reader_generator(
      params,
      dataset_fn=dataset_fn.pick_dataset_fn(params.file_type),
      decoder_fn=decoder.decode,
      parser_fn=parser.parse_fn(params.is_training))
      ....


  dataset = reader.read(input_context=input_context)
  return dataset
```


The class being responsible for building the input pipeline is
[InputReader](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/core/input_reader.py#L214)
with interface

```python
class InputReader:
  """Input reader that returns a tf.data.Dataset instance."""

  def __init__(
      self,
      params: cfg.DataConfig,
      dataset_fn=tf.data.TFRecordDataset,
      decoder_fn: Optional[Callable[..., Any]] = None,
      combine_fn: Optional[Callable[..., Any]] = None,
      sample_fn: Optional[Callable[..., Any]] = None,
      parser_fn: Optional[Callable[..., Any]] = None,
      filter_fn: Optional[Callable[..., tf.Tensor]] = None,
      transform_and_batch_fn: Optional[
          Callable[
              [tf.data.Dataset, Optional[tf.distribute.InputContext]],
              tf.data.Dataset,
          ]
      ] = None,
      postprocess_fn: Optional[Callable[..., Any]] = None,
  ):
  ....

  def read(self,
            input_context: Optional[tf.distribute.InputContext] = None,
            dataset: Optional[tf.data.Dataset] = None) -> tf.data.Dataset:
      """Generates a tf.data.Dataset object."""
      if dataset is None:
        dataset = self._read_data_source(self._matched_files, self._dataset_fn,
                                        input_context)
      dataset = self._decode_and_parse_dataset(dataset, self._global_batch_size,
                                              input_context)
      dataset = _maybe_map_fn(dataset, self._postprocess_fn)
      if not (self._enable_shared_tf_data_service_between_parallel_trainers and
              self._apply_tf_data_service_before_batching):
        dataset = self._maybe_apply_data_service(dataset, input_context)

      if self._deterministic is not None:
        options = tf.data.Options()
        options.deterministic = self._deterministic
        dataset = dataset.with_options(options)
      if self._autotune_algorithm:
        options = tf.data.Options()
        options.autotune.autotune_algorithm = (
            tf.data.experimental.AutotuneAlgorithm[self._autotune_algorithm])
        dataset = dataset.with_options(options)
      return dataset.prefetch(self._prefetch_buffer_size)
```

Therefore, customizing the input pipeline is equivalent to having customized
versions of `dataset_fn`, `decoder_fn`, etc. The execution order is generally
as:

```
dataset_fn -> decoder_fn -> combine_fn -> parser_fn -> filter_fn ->
transform_and_batch_fn -> postprocess_fn
```

The `transform_and_batch_fn` is an optional function that merges multiple
examples into a batch and its default behavior to `dataset.batch` if not
specified. In this workflow, the functions before `transform_and_batch_fn`, e.g.
`dataset_fn`, `decoder_fn`, consume tensors without the batch dimension, while
`postprocess_fn` will consume tensors with the batch dimension.

We have essentially covered
[decoder_fn](https://github.com/tensorflow/models/blob/master/official/vision/docs/read_custom_datasets.md#decoder),
and `parser_fn` is another very important one that takes the decoded raw tensors
dict and parses them into a dictionary of tensors that can be consumed by the
model. It will be executed after decoder_fn.

It is also worth noting that optimizing of the input pipeline through
batching, shuffling and prefetching is also implemented in this class.

## Parser

A custom data loader can also be useful if you want to take advantage of
features such as data augmentation.

Customizing preprocessing is useful because it allows the user to tailor the
preprocessing steps to suit the specific requirements of the task. While there
are standard preprocessing techniques that are commonly used, different
applications may require different preprocessing steps. Additionally, custom
preprocessing can also improve the efficiency and accuracy of the model by
removing unnecessary steps, reducing computational resources or adding steps
that are important to the specific task being addressed.

For example, tasks such as object detection or segmentation may require
additional preprocessing steps such as resizing, cropping, or data augmentation
to improve the robustness of the model. Below are some essential steps to
customize a parser.

### Instructions

*   **Create a Subclass**
    <br>

<dd><dl>

 Like Decoder, create `class Parser(parser.Parser)` in the same file.The
`Parser` class should be a childclass of the
[generic parser interface](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/parser.py)
and must implement all the abstract methods. It should have the implementation
of abstract methods `_parse_train_data` and `_parse_eval_data`, to generate
images and labels for model training and evaluation respectively. The below example
takes only two arguments but one can freely add as many arguments as needed.

```python
class Parser(parser.Parser):

 def __init__(self, output_size: List[int], num_classes: float):

   self._output_size = output_size
   self._num_classes = num_classes
   self._dtype = tf.float32

    ....
```

<br>

Refer to the data parser and processing [class](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/maskrcnn_input.py) for Mask R-CNN for more complex cases. The class has multiple parameters related to data augmentation, masking, anchor boxes, data type of output image and more.

</dd></dl>

<br>

*   **Complete Abstract Methods**<br>

<dd><dl> 

To define your own Parser, the user should override abstract functions
[_parse_train_data](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/vision/dataloaders/parser.py#L26)
and
[_parse_eval_data](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/vision/dataloaders/parser.py#L39)
of the
[parser](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/parser.py)
interface in the subclass, where decoded tensors are parsed with pre-processing
steps for training and evaluation respectively. The output from the two
functions can be any structure like a tuple, list or dictionary.

```python
  @abc.abstractmethod
  def _parse_train_data(self, decoded_tensors):
    """Generates images and labels that are usable for model training.

    Args:
      decoded_tensors: a dict of Tensors produced by the decoder.

    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """
    pass

  @abc.abstractmethod
  def _parse_eval_data(self, decoded_tensors):
    """Generates images and labels that are usable for model evaluation.

    Args:
      decoded_tensors: a dict of Tensors produced by the decoder.

    Returns:
      images: the image tensor.
      labels: a dict of Tensors that contains labels.
    """
    pass

```

The input of `_parse_train_data` and `_parse_eval_data` is a dict of Tensors
produced by the decoder; the output of these two functions is typically a tuple
of (processe_image, processed_label). The user may perform any processing steps
in these two functions as long as the interface is aligned. Note that the
processing steps in `_parse_train_data` and `_parse_eval_data` are typically
different since data augmentation is usually only applied to training. For
Example, refer to the
[Data parser](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/vision/dataloaders/classification_input.py#L166)
and processing steps for classification. We can observe that

<dd><dl>

-For `_parse_train_data`, the following steps are performed</dd></dl>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Image decoding<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Random cropping<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Random flipping<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Color jittering<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Image resizing<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Auto-augmentation with autoaug, randaug etc.<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Image normalization<br>

<dd><dl><dd><dl>

-For `_parse_eval_data`, the following steps are performed</dd></dl></dd></dl>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;
Image decoding<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Center cropping<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Image resizing<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Image normalization<br>

</dd></dl>

**Additional Methods**

The subclass (say sample_input.py) must include implementations for all of the
abstract methods defined in the Interface
[Decoder](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/decoder.py)
and
[Parser](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/parser.py)
, as well as any additional methods that are necessary for the subclass's
functionality.

For Example, In
[object detection](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/vision/dataloaders/tf_example_decoder.py#L72),
the decoder will take the serialized example and output a dictionary of tensors
with multiple fields that process and analyze to detect objects and determine
their location and orientation in the image. Separate methods for each of the
above fields can make the code easier to read and maintain, especially when the
class contains a large number of methods.

Refer
[Data parser](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/retinanet_input.py)
for Object Detection here.

### Example

Creating a Parser is an optional step and it varies with the use case. Below are
some use cases where we have included the Decoder and Parser based on the
requirements.

Use case |    Decoder/Parser |
-------------------------------------------------------------------------------------------------------------------------------------------------------- | ----
[Classification](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/classification_input.py) |  Both Decoder and Parser
[Segmentation](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/retinanet_input.py) |  Only Parser

## Input Pipeline

Decoder and Parser discussed previously define how to decode and parse per data
point e.g. an image. However a complete input pipeline would need to handle
reading data from files in a distributed system, applying random perturbations,
batching etc. You may find more details about these concepts
[here](https://www.tensorflow.org/guide/data_performance#optimize_performance).

We have established a well tuned input pipeline as defined in the [InputReader](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/core/input_reader.py#L214) class, such that the user won’t need to modify it in most cases. The input pipeline roughly follows<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Shuffling the files<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Decoding<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Parsing<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Caching<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;If training: repeat and shuffle<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Batching<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;- &nbsp;&nbsp;&nbsp;Prefetching<br>

For the rest of this section, we will discuss one particular use case that
requires the modification of the typical pipeline by maybe creating a subclass
of the
[InputReader](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/core/input_reader.py#L214).

### Combines multiple datasets

Create a custom InputReader by subclassing
[InputReader](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/core/input_reader.py#L214)
interface. Custom InputReader class allows the user to combine multiple
datasets, helps in mixing a labeled and pseudo-labeled dataset etc. The business
logic is implemented in the `read()` method which finally generates a
`tf.data.Dataset` object.

The exact implementation of an InputReader can vary depending on the specific
requirements of your task and the type of input data you're working with, data
format, and preprocessing requirements.

Here is an example of how to create a custom InputReader by subclassing
[InputReader](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/core/input_reader.py#L214)
interface:

```python
class CustomInputReader(input_reader.InputReader):

 def __init__(self,
              params: cfg.DataConfig,
              dataset_fn=tf.data.TFRecordDataset,
              pseudo_label_dataset_fn=tf.data.TFRecordDataset,
                ....):

 def read(
     self,
     input_context: Optional[tf.distribute.InputContext] = None
 ) -> tf.data.Dataset:


   labeled_dataset =   ....
   pseudo_labeled_dataset =   ....
   dataset_concat = tf.data.Dataset.zip(
       (labeled_dataset, pseudo_labeled_dataset))
  ....

   return dataset_concat.prefetch(tf.data.experimental.AUTOTUNE)

```

### Example

Refer to the
[InputReader](https://github.com/tensorflow/models/blob/b1a7752c5137822a32bd0dd70a0cb96e807ea411/official/vision/dataloaders/input_reader.py#L124)
for vision in TFM. The `CombinationDatasetInputReader` class mixes a labeled and
pseudo-labeled dataset and returns a `tf.data.Dataset` instance.
