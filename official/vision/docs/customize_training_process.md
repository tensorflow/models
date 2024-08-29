# Customize Training Process
## Overview

Customizing the training process allows users to tailor for the specific problem
they are trying to solve and the characteristics of the data they are working
with. This can be particularly important if we have a complex model architecture
and/or want to optimize for specific objectives. For example, we may want to use
custom loss functions or metrics to optimize for specific objectives or to
evaluate the performance of your model, tweaking the model architecture to
better fit the data.

## Customize train and validation step

### Instructions

We define all modeling artifacts of a particular machine learning task as a
`Task` object. The `Task` includes `build_model` for creating the model
instance, `build_inputs` for defining tf.data input pipeline, `train_step` and
`validation_step` for defining the computation logic, `build_metrics` for
streaming metrics etc.

Users has the flexibility to define and customize the training and validation
steps to fine-tune the training process to better fit the specific use case. An
example Task inherited from
[ImageClassificationTask](https://github.com/tensorflow/models/blob/master/official/vision/tasks/image_classification.py#L32)
can be found
[here](https://github.com/tensorflow/models/blob/master/official/vision/examples/starter/example_task.py).

*   The `train_step` typically encapsulates the logic of a forwardpass ,
    computing the gradients of the loss function with respect to the model's
    trainable variables, applying the gradients to update the model parameters
    via the optimizer, and returning the loss.

*   While the `validation_step` typically only runs the forward pass without
    updating the model weights.
    
    |       |      |
    |-----|-----|
    <font><pre><span style="color: #2E48CC;">def</span> train_step(<span style="color: #2E48CC;">self</span>,<br> inputs, <br> model: tf.keras.<span style="color: #52277E;">Model,</ span><br>optimizer: <br>tf.keras.optimizers.<span style="color: #52277E;">Optimizer,</span><br>metrics=<span style="color: #2E48CC;">None</span>):<br> <span style="color: #116644;">"""Does forward and backward. <br>With distribution strategies, <br>    this method runs on devices. <br><br>Args:<br> inputs: a dictionary of input tensors.<br> model: the model, forward pass<br>definition.<br> optimizer: the optimizer for<br>training step.<br> metrics: a nested structure of metrics <br>objects.<br><br>Returns: <br> A dictionary of logs."""<br></pre></font>|<font><pre><span style="color: #2E48CC;">def</span> validation_step(<span style="color: #2E48CC;">self</span>,<br>inputs, <br>model: tf.keras.<span style="color: #52277E;">Model,</span><br>     metrics=<span style="color: #2E48CC;">None</span>):<br> <span style="color: #116644;">"""Validation step. <br>With distribution strategies, <br>this method runs on devices. <br><br>Args:<br> inputs: a dictionary of input tensors.<br> model: the keras.Model.<br> metrics: a nested structure of metrics <br>objects.<br><br><Returns: <br> A dictionary of logs."""<br></pre></font>|
    
    
    

The arguments passed to the
[train_step](https://github.com/tensorflow/models/blob/master/official/core/base_task.py#L221)
are typically `inputs`, `model`, `optimizer` and `metrics`, whereas the
arguments passed to the
[validation_step](https://github.com/tensorflow/models/blob/master/official/core/base_task.py#L280)
are `model`, `metrics` and `inputs`. Note that the argument list is customizable
if needed.

*   `inputs` - it follows the output from data loader defined in build_inputs,
    and is typically a tuple of (features, labels). Other data structures, such
    as dictionaries, can also be used, as long as it is consistent between
    output from build_inputs and input used here.

*   `model` - the model is a `tf.keras.Model` that is built from build_model.
    Users can either choose from
    [TFMG models](https://github.com/tensorflow/models/tree/master/official/vision/modeling)
    based on their use cases, such as
    [classification model](https://github.com/tensorflow/models/blob/master/official/vision/modeling/classification_model.py)
    and
    [segmentation model](https://github.com/tensorflow/models/blob/master/official/vision/modeling/segmentation_model.py)
    or create their own custom model.

*   `optimizer` - During the `train_step` , users can use any optimizer that is
    available in the `tf.keras.optimizers` module or any custom optimizer that
    they have defined. When using mixed precision training, it is recommended to
    either use the `tf.keras.mixed_precision.LossScaleOptimizer` wrapper around
    the optimizer to scale the loss values to avoid underflow or the specified
    optimizer should be of the `tf.keras.mixed_precision.LossScaleOptimizer`
    type.

    During the `validation_step`, we generally do not need to use an `optimizer`
    because we are not updating the model weights based on the validation data.
    The goal of the validation step is to evaluate the model's performance on a
    separate set of data to check for overfitting and improve the model's
    generalization ability. Refer to TFM optimizers
    [here](https://github.com/tensorflow/models/tree/master/official/modeling/optimization).

*   `metrics` - The metrics is to evaluate the performance of the model during
    training and validation. Users can use either predefined or custom metrics.
    Predefined metrics from the `tf.keras.metrics` module are those that are
    available out of the box and are commonly used. We have also implemented
    common metrics in TF Model Garden. For example, the object detection models
    often use metrics such as mean average precision at different levels of
    intersection over union (IoU) to assess how accurately the model is
    detecting objects. Refer code :
    [Semantic Segmentation](https://github.com/tensorflow/models/blob/master/official/vision/tasks/semantic_segmentation.py#L193).

To implement a customized `train_step`, the following basic structure is
recommended:

*   Unpack the batch data into input features and target.
*   Perform a forward pass through the model using the input features to
    generate predicted outputs.
*   Calculate the loss between the predicted outputs and the target.
*   Update the model's trainable variables using the gradients.
*   (Optionally) Update any relevant metrics to evaluate the model's
    performance. The exact steps performed in the custom `train_step` will
    depend on the specific requirements and objectives of the task being
    performed.

Similarly, a custom `validation_step` function follows the similar steps with an
exception. In contrast to the `train_step`, the `validation_step` does not
update the model weights based on the gradients.

Here is an example of how to implement a custom training step:

```python
def train_step(self,
         inputs: Tuple[Any, Any],
         model: tf.keras.Model,
         optimizer: tf.keras.optimizers.Optimizer,
         metrics: Optional[List[Any]] = None) -> Mapping[str, Any]:

     #Unpack the batch data into input features and target
     features, labels = inputs
     ......

     # Perform a forward pass through the model using the input features to generate
     # predicted outputs.
     outputs = model(features, training=True)
     ......

     # Calculate the loss between the predicted outputs and the target.
    loss = self.build_losses(
         model_outputs=outputs, labels=labels, aux_losses=model.losses)
     ......

     # For mixed_precision policy, when LossScaleOptimizer is used, loss is scaled for
     # numerical stability.
     if isinstance(optimizer,
                    tf.keras.mixed_precision.LossScaleOptimizer):
       scaled_loss = optimizer.get_scaled_loss(scaled_loss)

    # Update the model's trainable variables using the gradients of the loss with respect
    # to the variables.
      tvars = model.trainable_variables
      grads = tape.gradient(scaled_loss, tvars)
     ......

    # Scales back gradient before apply_gradients when LossScaleOptimizer is used.
      optimizer.apply_gradients(list(zip(grads, tvars)))

    # update any relevant metrics to evaluate the model's performance
      logs = {self.loss: loss}
      if metrics:
         for metric in metrics:
             metric.update_state(labels, outputs)
      return logs
```

In this example, the `train_step` method is overridden with custom training
behavior. The method takes a batch of training data as input and performs a
forward pass to compute the predictions, calculates the loss, and updates the
model weights based on the gradients computed by the optimizer. Metrics are also
computed and updated during the process. Refer
[example_task.py](https://github.com/tensorflow/models/blob/master/official/vision/examples/starter/example_task.py#L127)
for complete code.

Here's an example code snippet that demonstrates how to create a custom
validation step:

```python
 def validation_step(self,
        inputs: Tuple[Any, Any],
        model: tf.keras.Model,
        metrics: Optional[List[Any]] = None) -> Mapping[str, Any]:

     #Unpack the batch data into input variables and target variables.
     features, labels = inputs

     # Perform a forward pass through the model using the input variables to generate
     # predicted outputs.
     outputs = model(features, training=True)

     # Calculate the loss between the predicted outputs and the target variables.
     loss = self.build_losses(
            model_outputs=outputs, labels=labels, aux_losses=model.losses)

     # Update relevant metrics to evaluate the model's performance on the validation data.
     logs = {self.loss: loss}
     if metrics:
        for metric in metrics:
            metric.update_state(labels, outputs)
     return logs
```

### Example

Tasks such as Image Classification and Semantic Segmentation are inherited from
Base Task. The `train_step` function represents the training step and the
`validation_step` function represents the validatation step in the task class.

## Customize writing summary data

### Instructions

In TFM, scalar summaries are simple numerical values that can represent various
metrics like loss, accuracy, or other performance indicators. The default
`eval_summary_manager` only writes scalar summaries. At times, it becomes
essential for writing more complex summaries beyond the default scalar
summaries. We aim to include custom information or messages , such as image
visualizations or additional metrics, into our summaries. To incorporate these
elements, we must customize the
[SummaryManager](https://github.com/tensorflow/models/blob/master/orbit/utils/summary_manager.py#L24).

NOTE : The SummaryManager modification required some engineering work. It’s good
for introducing a new format of summary like an image. It is unnecessary for
simple tasks like adding a new scalar summary.

A custom SummaryManager class can be defined using the
[SummaryManager](https://github.com/tensorflow/models/blob/master/orbit/utils/summary_manager.py#L24)
utility class or
[SummaryManagerInterface](https://github.com/tensorflow/models/blob/master/orbit/utils/summary_manager_interface.py#L20)
utility interface. This class contains functions for managing summary writing,
providing users the flexibility to define their own customized implementation
for these methods.

The custom SummaryManager class should implement the `flush()`,
`summary_writer()` and `write_summaries()` methods when implementing the
[SummaryManagerInterface](https://github.com/tensorflow/models/blob/master/orbit/utils/summary_manager_interface.py#L20).

Whereas, when creating a subclass of the
[SummaryManager](https://github.com/tensorflow/models/blob/master/orbit/utils/summary_manager.py#L24),
there's no requirement for users to implement all of the aforementioned methods
in their custom subclass. The methods you choose to implement will vary based on
your customization goals. Additionally, you can reuse methods already provided
by the
[SummaryManager](https://github.com/tensorflow/models/blob/master/orbit/utils/summary_manager.py#L24)
parent class to streamline your implementation.

*   `summary_writer()` - This method is used to retrieve the summary writer
    object associated with a specific subdirectory. It takes in the argument
    `relative_path` for writing summaries, relative to the summary directory.
    The default value is empty, representing the root directory.

*   `write_summaries()` - This method generates summaries based on the provided
    dictionary of values, i.e. `summary_dict`. This function iteratively
    generates subdirectories for any nested dictionaries present in
    `summary_dict`. As a result, a directory hierarchy is established, and is
    visualized in the TensorBoard as distinct colored curves.

*   `flush()` - This method is used to flush the summaries to the log file. It
    takes no arguments and returns no value. The `flush()` method is important
    because it ensures that all of the summaries are written to the log file
    even if the program crashes or is interrupted. This allows you to recover
    the summaries even if the program does not complete successfully.

The customized SummaryManager will be passed to the
[run_experiment](https://github.com/tensorflow/models/blob/master/official/vision/train.py#L58)
method in the
[launch script](https://github.com/tensorflow/models/blob/master/official/vision/train.py#L65).

The motivation of customizing a `SummaryManager` is typically due to the
requirement of having custom information to be collected. The summary is
typically generated and collected in the
[validation step](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py#L423-L429).
The outputs of the validation step will be further passed into
[aggregate_logs](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py#L444-L449),
which will eventually be aggregated through
[reduce_aggregated_logs](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py#L465-L470).
The outputs of
[reduce_aggregated_logs](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py#L465-L470)
will be collected by the summary manager to detect this information and
subsequently include it in the generated summary.

Additionally, The `save_summary` parameter within
[run_experiment](https://github.com/tensorflow/models/blob/master/official/core/train_lib.py#L316)
governs whether a summary is written to the designated folder. Orbit controller
writes the train outputs to a folder with a
[summary writer](https://github.com/tensorflow/models/blob/master/orbit/controller.py#L509).
It requires an
[eval_summary_manager](https://github.com/tensorflow/models/blob/master/orbit/controller.py#L325)
to write the evaluation summary.

Here's an example code snippet that demonstrates how to create a custom
SummaryManager and a number of methods that you can override to implement your
custom SummaryManager.

```python
class CustomSummaryManager(SummaryManagerInterface):

 def __init__(self, summary_dir, summary_fn, global_step=None):
   self._enabled = summary_dir is not None
   self._summary_dir = summary_dir
   self._summary_fn = summary_fn
   self._summary_writers = {}
   ......

 def summary_writer(self, relative_path=""):
   if self._summary_writers and relative_path in self._summary_writers:
     return self._summary_writers[relative_path]
   ......

   else:
     self._summary_writers[relative_path] = tf.summary.create_noop_writer()
   return self._summary_writers[relative_path]

 def flush(self):
   if self._enabled:
    tf.nest.map_structure(tf.summary.flush, self._summary_writers)

 def write_summaries(self, summary_dict):
   if not self._enabled:
     return

   for name, value in summary_dict.items():
     if isinstance(value, dict):
       self._write_summaries(
          value, relative_path=os.path.join(relative_path, name))
     else:
       with self.summary_writer(relative_path).as_default():
         self._summary_fn(name, value, step=self._global_step)
```

You can visualize the logged data (summaries) in TensorBoard to monitor the
training progress.

### Example

We have developed a class of custom summary manager that creates scalar and
image summary. The class
[ImageScalarSummaryManager](https://github.com/tensorflow/models/blob/master/official/vision/utils/summary_manager.py#L24)
inherits from the
[SummaryManager](https://github.com/tensorflow/models/blob/master/orbit/utils/summary_manager.py#L24)
class, which itself derives from the
[SummaryManagerInterface](https://github.com/tensorflow/models/blob/master/orbit/utils/summary_manager_interface.py#L20).

## Customize metrics

### Keras metrics

Custom metrics can be defined using the `tf.keras.metrics.Metric` class. This
class encapsulates metric logic and state, allowing the user to define their own
custom metrics or use one of the built-in metrics provided by TensorFlow. The
custom metrics class should implement the `__init__()`, `update_state()` and
`result()` methods, and call the parent constructor to initialize the metric
state.

*   `__init__()` - This method is called when the metric is first created. You
    can use this method to initialize the state variables for your metric.

*   `update_state()` - This method is used to update the state variables of the
    metric with new data. For instance, when computing the accuracy of a model,
    the user can update the metric's state for each batch of predictions using
    the `update_state(targets, predictions)` method. Here, `targets` represent
    the true labels, and `predictions` are the model's predicted labels. As we
    continue to update the metric's state throughout the training process, the
    metric will accumulate values that can be used to compute the final accuracy
    score.

*   `result()` - The result method is used to compute the final value of the
    custom metric after training. Refer to the example below.

Here's an example code snippet that demonstrates how to create a custom metric
and a number of methods that you can override to implement your custom metric.

```python
class MyMetric(tf.metrics.Metric):

  def __init__(self, name='my_metric'):
    super(MyMetric, self).__init__(name)
    self.total = tf.Variable(0.0, name='total')
    self.count = tf.Variable(0.0, name='count')

  def update_state(self, y_true, y_pred, sample_weight=None):
    self.total.assign_add(tf.reduce_sum(y_true * y_pred))
    self.count.assign_add(tf.reduce_sum(sample_weight))

  def result(self):
    return self.total / self.count

  def reset_states(self):
    self.total.assign(0.0)
    self.count.assign(0.0)
```

#### Example

Custom metrics can be used to evaluate the performance of the models on specific
tasks or objectives that may not be adequately captured by standard metrics like
accuracy or F1-score. Some of the task specific examples of custom metrics are
[InstanceMetrics](https://github.com/tensorflow/models/blob/master/official/vision/evaluation/instance_metrics.py)
and
[Segmentation_metrices](https://github.com/tensorflow/models/blob/master/official/vision/evaluation/segmentation_metrics.py)
instance detection & segmentation.

### Python-based Metrics

Apart from the metric built on Keras, users also have the option to create
metrics with even greater flexibility, Python-based metrics. The open-source
COCO Evaluation Metric serves as an illustration of Python-based metrics that
are implemented in Python.

Users can create a customized Python-based metric by either using
[COCOEvaluator.py](https://github.com/tensorflow/models/blob/master/official/vision/evaluation/coco_evaluator.py#L41)
as a guide to devise your metric or by creating a subclass of
[COCOEvaluator.py](https://github.com/tensorflow/models/blob/master/official/vision/evaluation/coco_evaluator.py#L41)
for creating new detection/segmentation metrics.

While crafting a custom Python-based metric class, ensure it encompasses metric
logic, state, evaluation mechanism, and result. Look into the potential methods
that should likely be incorporated into the class as indicated below:

*   `__init__()` - This function is used during the initial creation of the
    metric. You can utilize this function to set up and initialize the state
    variables required for your metric.
*   `update_state()` - This method is called when the metric is updated with new
    data. You need to use this method to update and aggregate detection results
    and ground-truth data.
*   `reset_states()`: This method is called to reset the metric's state
    variables. You can use this method to clear the metric's results.
*   `result()` - The result method is used to calculate the ultimate value of
    the customized metric once the training process is complete.

Refer to the example below to create your own Python-based metric.

```python
class CustomPythonEvaluator(object):

  def __init__(self,
               annotation_file,
               include_mask,
               need_rescale_bboxes=True,
               per_category_metrics=False,
               ......):

  def reset_states(self):
    self._predictions = {}
    if not self._annotation_file:
      self._groundtruths = {}

  def result(self):
    metric_dict = ......
    self.reset_states()
    return metric_dict

  def update_state(self, groundtruths, predictions):
    groundtruths, predictions =self._convert_to_numpy(groundtruths,
                                                       predictions)
    for k in self._required_prediction_fields:
     ......

    for k, v in six.iteritems(predictions):
      if k not in self._predictions:
        self._predictions[k] = [v]
      else:
        self._predictions[k].append(v)
     ......

    for k in self._required_groundtruth_fields:
     ......

    for k, v in six.iteritems(groundtruths):
      if k not in self._groundtruths:
        self._groundtruths[k] = [v]
      else:
        self._groundtruths[k].append(v)
     ......

```

Subsequently, users should incorporate this customized Python-based metric class
into their relevant tasks for constructing detection metrics. This also involves
capturing interim outcomes during the validation process and deriving conclusive
results once validation concludes.

For detailed guidance, refer to the
[build_metrics()](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py#L284)
method, the
[aggregate_logs()](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py#L432)
method, and the
[reduce_aggregated_logs()](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py#L458)
method within the
[retinanet task](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py).
These functions facilitate the integration of the custom metric, the computation
of interim outcomes during validation, and the final result computation after
validation.

It's worth noting that these computations are primarily executed on the CPU.

## Customize Loss

### Instructions

Customizing loss functions can be useful where the standard loss functions do
not accurately capture the performance of the model. To define a custom loss
function, users can define a function that should take the predicted output of
the model and the ground-truth labels as input tensors and return a tensor that
contains the loss value for each example in the batch.

To customize a loss function in TensorFlow, you can create a custom loss class
that inherits from the `tf.keras.losses.Loss` class. This
[Loss](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/losses.py#L48)
class provides `__init__()`and `call()` methods that you can override to
implement your custom loss function.

*   `init()` : This method is called when the loss is first created. You can use
    this method to initialize the state variables for your loss function.
*   `call()` : This method is called when the loss is evaluated with a new batch
    of data. You can use this method to calculate the loss for the current batch
    of data.

Here is an example of a custom loss class and a number of methods that you can
override to implement your custom loss.

```python
class CustomLoss(tf.keras.losses.Loss):

    def __init__(self,
                 Input_size,
                 alpha=0.25,
                 num_classes=10,
                 ......

                 cls_weight=0.3,
                 reduction=tf.keras.losses.Reduction.NONE,
                 name=None):
                 self._num_classes = num_classes
                 self._input_size = input_size
                 ......

                 super().__init__(reduction=reduction, name=name)


    def call(self, labels, predictions):
        positive_label_mask = tf.equal(labels, 1.0)
        cross_entropy = (tf.nn.sigmoid_cross_entropy_with_logits(
          labels=labels,logits=predictions))
        probs = tf.sigmoid(predictions)
        ......

        modulator = tf.pow(1.0 - probs, self._gamma)
        loss = modulator * cross_entropy
        weighted_loss = tf.where(positive_label_mask, self._alpha * loss,(
          1.0 - self._alpha) * loss)

        return weighted_loss
```

Once the custom loss function is implemented, it needs to be integrated into the
training loop. This involves modifying the training code to use the custom loss
function instead of the standard loss function provided by the framework.

### Example

In the
[retinanet task](https://github.com/tensorflow/models/blob/master/official/vision/tasks/retinanet.py)
definition, we use the custom loss in
[build_losses](https://github.com/tensorflow/models/blob/master/official/vision/tasks/maskrcnn.py#L254)
method. It calls the custom loss class
[focal_loss.py](https://github.com/tensorflow/models/blob/master/official/vision/losses/focal_loss.py).

