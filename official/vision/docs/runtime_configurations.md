# Runtime Configurations



## Available runtime configurations

In Model Garden,
[runtime configurations](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L140)
are a set of attributes used inside
[train_lib.py](https://github.com/tensorflow/models/blob/master/official/core/train_lib.py)
to ensure the training and/or evaluation jobs are properly configured for target
hardware and software environments. These attributes include e.g. the
[distribution strategy](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L169),
which controls how training is distributed across multiple devices; the
computation resources, which may control the
[number of GPUs](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L175)
or CPUs used for training. Runtime configurations are important to achieve
optimal performance and efficiency. A concrete example for running an image
classification task on TPU with `bfloat16`
[mixed_precision_dtype](https://www.tensorflow.org/guide/mixed_precision) can be
found
[here](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/image_classification/imagenet_mobilenetv1_tpu.yaml).

```python
runtime:
  distribution_strategy: 'tpu'
  mixed_precision_dtype: 'bfloat16'
task:
  ……
```

In this section, we would walk you through the available options, and we have
grouped them into three groups

*   Common parameters: configurations applicable for all hardware and software
    setup
*   TPU specific parameters: configurations applicable for TPU job only
*   GPU specific parameters: configurations applicable to GPU job only


#### Summary table

 | | |
|---- | ---|
|                 Common Parameters                  | [distribution_strategy](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L169)|                                      
 | | [mixed_precision_dtype](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L180)
 | | [loss_scale](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L181)
 | | [all_reduce_alg](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L178)
 | | [run_eagerly](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L182)
 | | [worker_hosts](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L176)
 | | [task_index](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L177)
 | | [enable_xla](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L170)
TPU Specific Params  | [tpu](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L174)
| |[tpu_enable_xla_dynamic_padder](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L197)
GPU Specific      | [num_gpus](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L175)
Others            | [gpu_thread_mode](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L171)
 | | [per_gpu_thread_count](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L173)
 | | [dataset_num_private_threads](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L172)
 | | [num_packs](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L179)


### Common Parameters

*   [distribution_strategy](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L169):
    *   Required parameter
    *   Default value: `'mirrored'`
    *   Data type: `String`

This parameter controls the exact
[tf.distribute.Strategy](https://www.tensorflow.org/guide/distributed_training)
used for setting up distributed training across multiple GPUs, multiple
machines, or TPUs. It allows users to easily distribute and parallelize their
training workloads across multiple machines, making it easier to scale up the
training process. Distributed training helps to reduce the time required to Note
that the `distribution_strategy` needs to be configured based on the target
software and hardware environment. software and hardware environment.

*   `tpu` distribution strategy: it lets you run your TensorFlow training on
    Tensor Processing Units (TPUs) through synchronous distributed training.
    TPUs provide their own implementation of efficient all-reduce and other
    collective operations across multiple TPU cores, which are used in `tpu`
    strategy.
*   `mirrored` distribution strategy: it implements synchronous training across
    multiple GPUs on one machine. It creates copies of all variables in the
    model on each device across all workers.
*   `multi_worker_mirrored` distribution strategy: this strategy implements
    synchronous distributed training across multiple workers, each with
    potentially multiple GPUs.
*   `parameter_server` distribution strategy: parameter server training is a
    common data-parallel method to scale up model training on multiple machines.
    A parameter server training cluster consists of workers and parameter
    servers. Variables are created on parameter servers and they are read and
    updated by workers in each step.

Note that the `distribution_strategy` needs to be configured based on the target
software and hardware environment.

*   [mixed_precision_dtype](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L180):
    *   Optional parameter
    *   Default value: `None`
    *   Data type: `String`

[Mixed precision](https://www.tensorflow.org/guide/mixed_precision) is the use
of both 16-bit and 32-bit floating-point types in a model during training to
make it run faster and use less memory. By keeping certain parts of the model in
the 32-bit types for numeric stability, the model will have a lower step time
and train equally as well in terms of the evaluation metrics such as accuracy.
The `mixed_precision_dtype` parameter is used to specify mixed precision policy,
and available options are:

*   `float32`
*   `float16`
*   `bfloat16` (TPU only)

If the `mixed_precision_dtype` is set to `tf.float16`, lower-precision dtypes
should be used whenever possible on those devices. However, variables and a few
computations should still be in float32 for numeric reasons so that the model
trains to the same quality. Modern accelerators can run operations faster in the
16-bit dtypes, as they have specialized hardware to run 16-bit computations and
16-bit dtypes can be read from memory faster

*   [loss_scale](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L181):
    *   Optional parameter
    *   Default value: `None`
    *   Data type: `String` or `Float`

This parameter specifies the type of loss scale, or 'float' value.
<span style="color: red;">This is used when setting the mixed precision
policy.</span>
[Loss scaling](https://www.tensorflow.org/guide/mixed_precision#loss_scaling) is
a process that multiplies the loss by a multiplier called the `loss scale`, and
divides each gradient by the same multiplier. Loss scaling can help avoid
numerical underflow in intermediate gradients when float16 tensors are used for
mixed precision training. By multiplying the loss, each intermediate gradient
will have the same multiplier applied. The most commonly used type is the
`dynamic` loss scale, where the loss scale will be dynamically updated over time
using an algorithm that keeps the loss scale at approximately its optimal value.

*   [all_reduce_alg](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L178):
    *   Optional parameter
    *   Default value: `None`
    *   Data type: `String`

It is used to specify the algorithm used to perform the all-reduce operation,
which is used to synchronize variables across multiple machines. For `mirrored`
strategy, valid values are `nccl` and `hierarchical_copy`. For
`multi_worker_mirrored` Strategy, valid values are `ring` and `nccl`. If None,
Distribution Strategy will choose based on device topology.

*   [run_eagerly](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L182):
    *   Required parameter
    *   Default value: `False`
    *   Data type: `Boolean`

The boolean parameter decides whether or not to perform the experiment eagerly.
If it is set to `False`, the training and evaluation logics will not be wrapped
in a `tf.function`. It is recommended to leave this as `False` unless your logic
cannot be run inside a `tf.function`, or you would like to perform step by step
debugging.

*   [worker_hosts](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L176):
    *   Optional parameter
    *   Default value: `None`
    *   Data type: `String`

`worker_hosts` is a parameter used to specify the network addresses of the
worker nodes in a distributed training setup. This variable is typically used
when performing multi-worker training with the TensorFlow distributed strategy.
The variable should be set to a comma-separated list of the worker nodes in the
form of 'host1:port,host2:port.

Example : worker_hosts: `$HOST1:port,$HOST2:port` - $HOST1 and $HOST2 are the IP
addresses of the hosts, and port can be chosen from any free port on the hosts.
Only the first host will write TensorBoard Summaries and save checkpoints.

*   [task_index](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L177):
    *   Optional parameter
    *   Default value: `-1`
    *   Data type: `Int`

`task_index` is a parameter typically used when performing multi-worker training
with the TensorFlow distributed strategy. It is used to specify the index of the
worker node in the network. Setting the task index variable is important, as the
index is used to keep track of the worker nodes in the network and ensure that
each worker is performing its assigned tasks correctly. For example,
worker_hosts: `$HOST1:port,$HOST2:port`, you have task_index: 0 on the first
host and task_index: 1 on the second and so on.

*   [enable_xla](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L170):
    *   Required parameter
    *   Default value: `False`
    *   Data type: `Boolean`

`enable_xla` is to enable or disable the XLA compiler in TensorFlow. The XLA
compiler is a just-in-time optimized compiler that can improve the performance
of TensorFlow models. XLA performs compiler optimizations, such as fusion, and
attempts to emit more efficient code. This may drastically improve the
performance. If set to `True`, the whole function needs to be compilable by XLA,
or an `errors.InvalidArgumentError` is thrown. If `None` (default), compiles the
function with XLA when running on TPU and goes through the regular function
execution path when running on other devices.

### TPU Specific Parameters

*   [tpu](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L174):
    *   Optional parameter
    *   Default value: `None`
    *   Data type: `String`

The String that represents the TPU address to connect to, if any. Must not be
None if `distribution_strategy` is set to `tpu`.

*   [tpu_enable_xla_dynamic_padder](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L197):
    *   Optional parameter
    *   Default value: `None`
    *   Data type: `Boolean`

It is an optional Boolean parameter in the TensorFlow runtime configuration. It
is used to enable dynamic padding for XLA (Accelerated Linear Algebra)
operations. XLA performs compiler optimizations, such as fusion, and attempts to
emit more efficient code. This may drastically improve the performance. If set
to `True`, the whole function needs to be compilable by XLA, or an
`errors.InvalidArgumentError` is thrown. If `None` (default), compiles the
function with XLA when running on TPU and goes through the regular function
execution path when running on other devices.

### GPU Specific Parameters

*   [num_gpus](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L175):
    *   Required parameter
    *   Default value: `0`
    *   Data type: `Int`

This is an attribute to specify the number of GPUs to use at each worker with
the distribution strategies. Note that with default value 0, the training
process won't utilize any GPU even if they are present.

In addition to the above parameters, we support more but less commonly used
parameters such as
[gpu_thread_mode](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L171),
[per_gpu_thread_count](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L173)
,
[dataset_num_private_threads](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L172)
and
[num_packs](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L179)
used for optimizing performance on GPU. Refer gpu performance
[guide](https://www.tensorflow.org/guide/gpu_performance_analysis).

Note: They are used in the TF environment variables
[here](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/utils/misc/keras_utils.py#L190).
But it requires to manually call `keras_utils.set_gpu_thread_mode_and_count`
parameter. So far, only legacy code, benchmark, and code from other parties call
them. Thus they are not automatically used and do not have effect when set
without calling `keras_utils.set_gpu_thread_mode_and_count`.

Please check
[here](https://github.com/tensorflow/models/blob/ea1054c5885ad8b8ff847db02c010f8b51e25f5b/official/core/config_definitions.py#L140)
for the full list of parameters.

## How to set runtime configurations

This section of the user guide illustrates some of the most common use cases on
how to set the runtime configurations. The most common configurations include
setting the device to use for training, setting the optimizer and the loss
function, setting the metric to use for evaluation, the number of workers, and
setting the distribution strategy.

Additionally, there may be other configuration settings to fine-tune the model
performance, such as the number of training epochs, batch size, learning rate,
weight decay, the learning rate decay, and the gradient clipping and more.

Below we list a few most commonly encountered use cases for user reference.

### Training on TPU

*   `mixed_precision_dtype`: **bfloat16**
    (<span style="color: red;">Recommended</span>)

```python
runtime:
  distribution_strategy: 'tpu'
  mixed_precision_dtype: 'bfloat32'
task:
  train_data:
    is_training: true
    global_batch_size: 4096
    dtype: 'bfloat32'
  validation_data:
    is_training: false
    global_batch_size: 4096
    dtype: 'bfloat32'
    drop_remainder: false
    ....
```

Please refer to this
[config file](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/image_classification/imagenet_mobilenetv1_tpu.yaml)
for a full example of running image classification with `bfloat16`
mixed_precision_dtype and `tpu` distribution_strategy.

*   `mixed_precision_dtype`: **float32**

```python
runtime:
  distribution_strategy: 'tpu'
  mixed_precision_dtype: 'float32'
task:
  train_data:
    is_training: true
    global_batch_size: 4096
    dtype: 'float32'
  validation_data:
    is_training: false
    global_batch_size: 4096
    dtype: 'float32'
    drop_remainder: false
    ....
```

Please refer to this
[config file](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/semantic_segmentation/deeplabv3plus_resnet101_cityscapes_tfds_tpu.yaml)
for a full example of running semantic segmentation with `float32`
mixed_precision_dtype and `tpu` distribution_strategy.

### Training on GPU

*   `mixed_precision_dtype`: **float16**
    (<span style="color: red;">Recommended</span>)

```python
runtime:
  distribution_strategy: 'mirrored'
  num_gpus: 4
  mixed_precision_dtype: 'float16'
  loss_scale: 'dynamic'
task:
  ……
  train_data:
    is_training: true
    global_batch_size: 4096
    dtype: 'float16'
  validation_data:
    is_training: false
    global_batch_size: 4096
    dtype: 'float16'
    drop_remainder: false
    ……
```

Please refer to this
[config file](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/image_classification/imagenet_mobilenetv2_gpu.yaml)
for a full example of image classification with `float16` mixed_precision_dtype
and `mirrored` distribution_strategy.

*   `mixed_precision_dtype`: **float32**

```python
runtime:
  distribution_strategy: 'mirrored'
  num_gpus: 4
  mixed_precision_dtype: 'float32'
  loss_scale: 'dynamic'
task:
  ……
  train_data:
    is_training: true
    global_batch_size: 4096
    dtype: 'float32'
  validation_data:
    is_training: false
    global_batch_size: 4096
    dtype: 'float32'
    drop_remainder: false
    ……
```

Please refer to this
[config file](https://github.com/tensorflow/models/blob/master/official/projects/pruning/configs/experiments/image_classification/imagenet_mobilenetv2_pruning_gpu.yaml)
for a full example of image classification with `float32` mixed_precision_dtype
and `mirrored` distribution_strategy.

## How to adjust according to different runtime configurations

While tuning runtime configurations of your job, it is important to be aware
that some task related configurations should be adjusted accordingly as well.
For example, if the number of accelerators is reduced, the `batch_size` should
be reduced accordingly, otherwise each accelerator will be allocated
proportionally more data.

Below are some commonly encountered use cases for reference.

### Reduce number of accelerators

Consider a use case , if the template YAML uses 8 GPUs for training but the user
has only 4 GPUs, it is recommended to follow the tips below. This will help
ensure that the model is trained as efficiently as possible and will help avoid
performance issues due to limited GPU resources.

*   Reduce batch size
*   Increase number of steps for train and validation
*   Modify learning_rate schedule
*   Decrease learning rate

### Increase number of accelerators

If we want to increase the number of accelerators, the adjustment will be the
opposite of the case above.

*   Increase batch size
*   Reduce number of steps for train and validation
*   Modify learning_rate schedule
*   Increase learning rate

We have provided a concrete example below for image classification on ImageNet
with `batch_size` to be 2048 and 4096:

 | | |
|---- | ---|
<font size="1"> <pre>global_batch_size: 4096<br> trainer:<br>  train_steps: <span style="color: #006666;">156000</span>  <span style="color: #880000;"># 500 epochs </span><br>  validation_steps: <span style="color: #006666;">13</span><br>  validation_interval: <span style="color: #006666;">312</span><br>  steps_per_loop: <span style="color: #006666;">312</span>   <span style="color: #880000;"># NUM_EXAMPLES <br>(1281167) // global_batch_size</span><br>  summary_interval: <span style="color: #006666;">312</span><br>  checkpoint_interval: <span style="color: #006666;">312</span><br>  optimizer_config:<br>    learning_rate:<br>      type: <span style="color: #008800;">'exponential'</span><br>      exponential:<br>        initial_learning_rate: <span style="color: #006666;">0.256 </span><span style="color: #880000;"># 0.008 <br>* batch_size / 128</span><br>        decay_steps: <span style="color: #006666;">780</span>  <span style="color: #880000;"># 2.5 * <br>steps_per_epoch</span><br>        decay_rate: <span style="color: #006666;">0.94</span><br>        staircase: <span style="color: #000088;">true.   </span><br>    warmup:<br>      type: <span style="color: #008800;">'linear'</span><br>      linear:<br>        warmup_steps: <span style="color: #006666;">1560</span></pre></font> | <pre>global_batch_size: 2048<br> trainer:<br>  train_steps: <span style="color: #006666;">312000  </span><span style="color: #880000;"># 500 epochs</span><br>  validation_steps: <span style="color: #006666;">26</span><br>  validation_interval: <span style="color: #006666;">624   </span><br>  steps_per_loop: <span style="color: #006666;">624</span>  <span style="color: #880000;"># NUM_EXAMPLES <br>(1281167) // global_batch_size</span><br>  summary_interval: <span style="color: #006666;">624</span><br>  checkpoint_interval: <span style="color: #006666;">624</span><br>  optimizer_config:<br>    learning_rate:<br>      type: <span style="color: #008800;">'exponential'</span><br>      exponential:<br>        initial_learning_rate: <span style="color: #006666;">0.128 </span><span style="color: #880000;"># 0.008 <br>* batch_size / 128</span><br>        decay_steps: <span style="color: #006666;">1560  </span><span style="color: #880000;"># 2.5 * <br>steps_per_epoch</span><br>        decay_rate: <span style="color: #006666;">0.94</span><br>        staircase: <span style="color: #000088;">true    </span><br>    warmup:<br>      type: <span style="color: #008800;">'linear'</span><br>      linear:<br>        warmup_steps: <span style="color: #006666;">3120</span></pre>

### Switch from GPU to TPU

Switching from GPU to TPU will allow users to take advantage of the TensorFlow
TPU distribution strategy, which in turn allows you to run your models on Users
may follow below suggestions to better take advantage of the TPU's strengths:

*   `float16` need to be changed to `bfloat16`
*   dtype of `train_data` and `validation_data` should be modified
*   May increase batch_size since TPU is more powerful
*   The batch size of any model should always be at least 64 (8 per TPU core),
    since the TPU always pads the tensors to this size. The ideal batch size
    when training on the TPU is 1024 (128 per TPU core), since this eliminates
    inefficiencies related to memory transfer and padding.

Refer config comparison of
[TPU](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/image_classification/imagenet_mobilenetv1_tpu.yaml)
and
[GPU](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/image_classification/imagenet_mobilenetv2_gpu.yaml)
using Image Classification examples below:


 | | |
|---- | ---|
<font><pre>runtime:<br>   distribution_strategy: <span style="color: #116644;">'mirrored'</span><br>   <b>mixed_precision_dtype: <span style="color: #116644;">'float16'</span> </b> <br>   loss_scale: <span style="color: #116644;">'dynamic'</span><br>task:<br>  ……<br>  train_data:<br>    ……<br>    <b>global_batch_size: <span style="color: #116644;">1024</span><br>    dtype: <span style="color: #116644;">'float16'</span> </b> <br><br>  validation_data:<br>    ……<br>    <b>global_batch_size: <span style="color: #116644;">1024</span><br>    dtype: <span style="color: #116644;">'float16'</span> </b> <br>……</pre></font> | <pre>runtime:<br>   distribution_strategy: <span style="color: #116644;">'tpu'</span><br>   <b>mixed_precision_dtype: <span style="color: #116644;">'bfloat16'</span> </b> <br><br>task:<br>  ……<br>  train_data:<br>    ……<br>    <b>global_batch_size: <span style="color: #116644;">4096</span><br>    dtype: <span style="color: #116644;">'bfloat16'</span> </b> <br><br>  validation_data:<br>    ……<br>    <b>global_batch_size: <span style="color: #116644;">4096</span><br>    dtype: <span style="color: #116644;">'bfloat16'</span> </b> <br>……</pre>|
