# Frequently Asked Questions


## FAQs of TF-Vision

--------------------------------------------------------------------------------

### Q1: How to get started with Tensorflow Model Garden TF-Vision?

This
[user guide](https://github.com/tensorflow/models/blob/master/official/README.md)
is a walkthrough on how to train and fine-tune models, and perform
hyperparameter tuning in TF-Vision. For each model/task supported in TF-Vision,
please refer to the corresponding tutorial to get more detailed instructions.

--------------------------------------------------------------------------------

### Q2: How to use the models under tensorflow_models/official/vision/?

*   **Available models under TF-Vision:** There is a good collection of models
    available in TF-Vision for various vision tasks: image classification,
    object detection, video classification, semantic segmentation and Instance
    segmentation. Please check
    [this page](https://github.com/tensorflow/models/blob/master/official/README.md)
    to know more about our available models. We will keep adding new supports,
    and your suggestions are appreciated.

*   **Fine-tune from a checkpoint:** TF-Vision supports loading pretrained
    checkpoints for fine-tuning. It can be simply done by specifying
    `task.init_checkpoint` and `task.init_checkpoint_modules` in the task
    configuration. The value of `task.init_checkpoint_modules` depends on the
    pretrained modules implementation which in general can be e.g. **all**,
    **backbone**, and/or **decoder** (for detection and segmentation). If set to
    **all**, all weights from the checkpoint will be loaded. If set to backbone,
    only weights in the **backbone** component will be loaded and other weights
    will be initialized from scratch. An example yaml file can be found
    [here](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/semantic_segmentation/deeplabv3plus_resnet101_cityscapes_tfds_tpu.yaml).

*   **Export SavedModel for serving:** To export any TF 2.x models we trained,
    including `tf.keras.Model` and the plain `tf.Module`, we use the
    `tf.saved_model.save()` API. Our
    [exporting library](https://github.com/tensorflow/models/tree/master/official/vision/serving/)
    offers functionalities to export SavedModel for CPU/GPU/TPU serving.
--------------------------------------------------------------------------------

### Q3: How to fully/partially load pretrained checkpoints (e.g. backbone) to perform transfer learning using TF-Vision?

TF-Vision supports loading pretrained checkpoints for fine-tuning. It can be
simply done by specifying `task.init_checkpoint` and
`task.init_checkpoint_modules` in the task configuration. The value of
`task.init_checkpoint_modules` depends on the pretrained modules implementation
which in general can either be e.g. all, backbone, and/or decoder (for detection
and segmentation). If set to all, all weights from the checkpoint will be
loaded. Let’s use a concrete example for elaboration. Suppose the requirements
are:

1.  Train a classification model with 10-class.

2.  save off the checkpoint of the model from step `1` ( but only save the
    backbone before the last Conv2D + softmax).

3.  use the checkpoint from step `2` to train a new classification model with 4
    novel classes.

For `2`, the model needs to specify the components to be saved in the checkpoint
in the
[checkpoint_items](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/modeling/classification_model.py#L119).
For `3`, you can specify the `init_checkpoint` and the `init_checkpoint_modules
='backbone'`. Then the new model with 4 classes will only initialize the
[backbone](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/tasks/image_classification.py#L80)
so that you can finetune the head. In this
[example](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/modeling/classification_model.py#L69),
backbone is everything before the global average pooling layer for the
classification model.

--------------------------------------------------------------------------------

### Q4: How to export the tensorflow models trained using the TF-Vision package?

To export any TF 2.x models we trained, including `tf.keras.Model` and the plain
tf.Module, we use the `tf.saved_model.save()` API. Our
[exporting library](https://github.com/tensorflow/models/tree/master/official/vision/serving)
offers functionalities to export SavedModel for CPU/GPU/TPU serving. Moreover,
with the exported SavedModel, it is possible to further convert it to a TFLite
model for on-device inference.

--------------------------------------------------------------------------------

### Q5: Where can I look for a config file and documentation for the TF-Vision pretrained models?

TF-Vision modeling library provides a collection of baselines and checkpoints
for various vision tasks including e.g. image classification, object detection,
video classification and segmentation. The supported pretrained models and
corresponding config file can be found
[here](https://github.com/tensorflow/models/blob/master/official/vision/MODEL_GARDEN.md).
Since we are actively developing new models, you are also recommended to check
our
[repository](https://github.com/tensorflow/models/tree/master/official/vision/configs/experiments/)
to find anything that has been added but not reflected in the documentation yet.

--------------------------------------------------------------------------------

### Q6: How to train a custom model for TF-Vision using models/official/vision?

We have provided an example
[project](https://github.com/tensorflow/models/blob/master/official/vision/examples/starter/README.md)
to demonstrate how to use TF Model Garden's building blocks to implement a new
vision project from scratch. All the internal/external projects built on top of
TFM can be found
[here](https://github.com/tensorflow/models/tree/master/official/projects/) for
reference.

--------------------------------------------------------------------------------

### Q7: How to profile flops? Looking for a template code on profiling FLOPs on a tf2 saved model. Any suggestions?

Set `log_model_flops_and_params` to true when exporting a saved model to log
`params` and `flops` as
[here](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/serving/export_saved_model.py#L80).

--------------------------------------------------------------------------------

### Q8: Turning on regenerate_source_id in the mask_r_cnn data pipeline would slow down the input pipeline?

The `regenrate_source_id` will add some extra
[computation](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/dataloaders/tf_example_decoder.py#L158)
but rarely create the bottleneck. You can do a POC to see if the input pipeline
is the bottleneck or not.

--------------------------------------------------------------------------------

### Q9: Are pre-trained models trained without any data preprocessing (e.g. mean, variance, or [-1, 1]) , i.e. they expect inputs in the range 0.0, 255.0?

All the pre-trained models are trained with well structured input pipelines
defined in [data loaders](https://github.com/tensorflow/models/tree/master/official/vision/dataloaders),
which typically includes e.g. normalization and augmentation. The normalization
approach used is task dependent, and you are recommended to check each task’s
corresponding input pipeline for confirmation:

*   Classification:
    [classification_input.py](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/classification_input.py).
*   Object Detection and Instance Segmentation:
    [maskrcnn_input.py](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/maskrcnn_input.py)
    and
    [retinanet_input.py](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/retinanet_input.py).
*   Semantic Segmentation:[segmentation_input.py](https://github.com/tensorflow/models/blob/master/official/vision/dataloaders/segmentation_input.py).

For example, the mean and std normalization is applied for classification tasks
by default.

--------------------------------------------------------------------------------

### Q10: How does the model garden library write a summary? How to add image summary?

Here are the general steps to write a summary:

*   The `save_summary` argument of `run_experiment` controls whether or not to
    write a summary to the folder
    [[ref](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/core/train_lib.py#L316)].
*   Orbit controller writes the train/eval outputs to a folder with a summary
    writer
    [[ref](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/orbit/controller.py#L327)].
    *   It requires an `eval_summary_manager` to write the summary
        [[ref](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/orbit/controller.py#L318)].
        The default `eval_summary_manager` only write scalar summary.

We have supported writing image summary to show predicted bounding boxes for
RetinaNet task. It can be adapted to write other types of summary. Here are the
steps:

*   We have created a custom summary manager that can write image summary
    [[ref](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/utils/summary_manager.py#L24)].

*   We optionally build the summary manager if the corresponding task is
    supported to write such summary
    [[ref](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/train.py#L65)],
    and pass it into the trainer as `eval_summary_manager`
    [[ref](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/core/train_lib.py#L322)].

*   In the task, we collect necessary predictions
    [[ref](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/tasks/retinanet.py#L423C1-L429C8)]
    in `validation_step`, update them in `aggregate_logs`
    [[ref](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/tasks/retinanet.py#L444C1-L449C77)],
    and add visualization into returned logs in `reduce_aggregated_logs`
    [[ref](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/tasks/retinanet.py#L465C4-L470C38)],
    so that the summary manager can identify such information and write it to
    summary.

*   We also need to set `allow_image_summary` to True in task config to enable
    this
    [[ref](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/core/config_definitions.py#L304)].



--------------------------------------------------------------------------------

### Q11: ViT Model: Running inference second time throws OOM error using the ViT model in inference only mode inside a colab with some modifications. It seems like we can only run inference once with it. The second time an input is fed, even if it's the same image, it runs out of GPU memory.

Check if there are any large intermediate tensors or objects that are still
alive from the previous inference. If you have any python variables that refer
to those tensors, then delete them. Also, you can import gc, and run garbage
collection through the command `gc.collect()`.

--------------------------------------------------------------------------------


### Q12: Is there a way to add a post train_step process similar to aggregate_logs and reduce_aggregated_logs for the validation step? How to include the individual training losses i.e. L = L_1 + L_2 + ... + L_n as part of the plots?

To do this, you will need to create a custom trainer. And to include the
individual training losses, you will need to create a `Mean` metric for each of
the losses and then propagate loss value to this metric during the train step.
Indeed it depends whether you need to run these metrics on CPU, if not, you can
do alike maskrcnn:
[define losses reference](https://github.com/tensorflow/models/blob/72d04629491e74c720e6414a52e16147aea75e41/official/vision/tasks/maskrcnn.py#L290C1-L298C42)
and
[define metrics reference](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/official/vision/tasks/maskrcnn.py#L339C1-L351C24).

Individual training losses should show up on Tensorboard if added in returned
logs. Average precision is reported in reduce_aggregated_logs.

--------------------------------------------------------------------------------

### Q13: How to run task.eval_step (or task.train_step) in eager mode?

You can add `tf.config.run_functions_eagerly = True` in the main function to
enable eager mode. Refer
[code](https://github.com/tensorflow/tensorflow/blob/9ec6201b4fc4a936210346b8c7b3f631117e4fbf/tensorflow/python/eager/polymorphic_function/polymorphic_function.py#L394)
here.

--------------------------------------------------------------------------------

### Q14: Does TFM support computing and reporting eval metrics separately on each dataset or should each custom task figure out how to do it?

Please find experiment config for single-task training and multi-task evaluation
[here](https://github.com/tensorflow/models/blob/c835649f62994af402c86caab202449a6c8e2f49/official/modeling/multitask/configs.py#L92).


--------------------------------------------------------------------------------

### Q15: An experiment ran 30k steps and the user wants to run ~10k more steps starting from where he left off. What's the recommended way to do this? Does he need to run a new job for 10k train_steps, with the init checkpoint set to the last checkpoint of the previous run?

If your previous training is complete with 30k and you want to train an
additional 10k, there are below ways:

*   set `init_checkpoint` to the last saved checkpoint
*   set `model_dir` to the training directory

Please be alert with the optimizer config. After you modify the training steps,
the LR curve will change.

Also, if you start the training in the same model dir, you will lose checkpoints
for the previous training run since we only keep the last 5. So if you are
planning to experiment with fine-tuning, it is suggested to start a new run.

Check out these
[configs](https://github.com/tensorflow/models/blob/c835649f62994af402c86caab202449a6c8e2f49/official/core/config_definitions.py#L273)
for storing the best checkpoint.

--------------------------------------------------------------------------------


### Q16: Does TF-Vision support multi workers with multi GPUs?

The prerequisite is to configure "MultiWorkerMirroredStrategy". The
`tf.distribute.MultiWorkerMirroredStrategy` implements synchronous distributed
training across multiple workers, each with potentially multiple GPUs. It
creates copies of all variables in the model on each device across all workers.
Please follow the guidelines
[here](https://www.tensorflow.org/guide/distributed_training).

--------------------------------------------------------------------------------

### Q17: When running multiple eval jobs with training jobs and modifying the model architecture under the `task` in the config yaml file using [MultiEvalExperimentConfig](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/configs.py), the eval jobs fail when loading the model. Is this an expected behavior ?

No, this is not expected behavior. The reason for the issue is that the eval
jobs are not reading the model architecture under the `task` config but from a
`eval_task` copy to reconstruct the model.

To address this issue, refrain from using the `eval_task` model
configurations.The model should be constructed from the `task`. The
[MultiTaskEvaluator](https://github.com/tensorflow/models/blob/c835649f62994af402c86caab202449a6c8e2f49/official/modeling/multitask/evaluator.py#L36)
class takes the eval data tasks and the model should be created
[here](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/train_lib.py).

--------------------------------------------------------------------------------

### Q18: What is the advised approach for determining whether it is in the training phase within the Task.build_losses() method?

Users can add a training argument in the
[build_losses()](https://github.com/tensorflow/models/blob/c835649f62994af402c86caab202449a6c8e2f49/official/core/base_task.py#L169)
method. build_losses is invoked in either from
[train_step](https://github.com/tensorflow/models/blob/c835649f62994af402c86caab202449a6c8e2f49/official/core/base_task.py#L251)
or
[validation_step](https://github.com/tensorflow/models/blob/c835649f62994af402c86caab202449a6c8e2f49/official/core/base_task.py#L300),
you can pass correct training arguments from each step.

--------------------------------------------------------------------------------

### Q19: How to mix two input datasets with fixed ratio in image classification training?

We have the implementation to support sampling from multiple training dataset
for all major tasks such as classification, retinanet, maskrcnn and segmentation
tasks. The `create_combine_fn` of
[input_reader.py](https://github.com/tensorflow/models/blob/c835649f62994af402c86caab202449a6c8e2f49/official/vision/dataloaders/input_reader.py#L47)
creates and returns a `combine_fn` for dataset mixing and is called in the
[build_inputs](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/official/vision/tasks/image_classification.py#L155)
method of the respective
[tasks](https://github.com/tensorflow/models/tree/master/official/vision/tasks).



Refer sample config below:

```yaml
train_data:
    input_path:
    d1: train1*,
    d2: train2*,
    weights:
    d1: 0.8
    d2: 0.2
```

--------------------------------------------------------------------------------

### Q20: How to add gradient magnitude logging to the metrics reported to TensorBoard if training from scratch using a model like mobilenet_imagenet?

You can add gradient magnitude logging into your metric log in the task class as
a new dictionary key-value pair.

Please refer to the
[Image Classification Task](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/official/vision/tasks/image_classification.py#L359C2-L366C55),
here you can obtain the pair of `gradient` and `trainable_variables` (grads,
tvar), add gradient magnitude and update the
[metric logs](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/official/vision/tasks/image_classification.py#L378).
It will be then processed in summary_manager’s
[write_summaries](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/orbit/controller.py#L540)
method.

--------------------------------------------------------------------------------

### Q21: Does TFM support computing and reporting eval metrics separately on each dataset or should each custom task figure out how to do it?

Please find experiment config for single-task training and multi-task evaluation
[here](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/official/modeling/multitask/configs.py#L92).

--------------------------------------------------------------------------------


### Q22: I am training the new YOLOv7 model on my own dataset. But encountered OOM in tpu_worker after approximately 6k steps. Whereas with the COCO dataset, it works fine. How to debug this OOM issue?

Add `prefetch_buffer_size` in the config file. A known issue exists regarding
the auto-tuning of the `prefetch_buffer_size`. You might consider setting a
suitable value explicitly instead. Know more about `prefetch_buffer_size`
[here](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch).
Refer below Example.

```yaml
train_data:
global_batch_size: 4096
dtype: 'bfloat16'
prefetch_buffer_size: 8
input_path: 'Input Path'
validation_data:
global_batch_size: 32
...
```

--------------------------------------------------------------------------------

### Q23: Is there a way to export a TF Model Garden model with arbitrary shape?

The user can set `input_image_size` to none if the model itself can be built
with arbitrary image size.

Refer below Example.

```python
export_saved_model_lib.export_inference_graph(
input_type='image_tensor',
batch_size=1,
input_image_size=[None, None],
params=exp_config,
checkpoint_path=tf.train.latest_checkpoint(model_dir),
export_dir=export_dir)

```

--------------------------------------------------------------------------------

### Q24: What is the number of images the model (for e.g. maskrcnn with resnet fpn) sees during training?

The number of images that is seen during training is train_steps *
global_batch_size. The relationship between global_batch_size and train_steps
can be explained as follows:

```python
train_epochs = 400
train_steps = math.floor(train_epochs * num_train_examples/train_data.global_batch_size)

// steps_per_loop = steps_per_epochs
steps_per_epoch = math.floor(num_train_examples/train_data.global_batch_size)
validation_steps = math.floor(num_val_examples/validation_data.global_batch_size)

// number of training steps to run between evaluations.
validation_interval = steps_per_epoch
```

Assuming a train dataset with `num_train_examples` images, and validation
dataset with `num_val_examples` images and the `train_epochs` is a
hyperparameter that you need to choose. `train_steps` depends on `train_epochs`
and `num_train_examples`.


--------------------------------------------------------------------------------

### Q25: Is there an early stopping option in Model Garden? Is there any documentation, or an example config?

Early stopping is not currently integrated into the Model Garden. An alternative
approach is to set up the training pipeline to export the best model based on
your specified criteria. The
[NewBestMetric](https://github.com/tensorflow/models/blob/d2427a562f401c9af118e47af2f030a0a5599f55/orbit/actions/new_best_metric.py#L31)
class keeps track of the best metric value seen so far. Subsequently, you can
train for an ample duration, and if signs of overfitting become apparent, you
have the flexibility to halt the run accordingly. That works well for one-off
experiments.

The `best_checkpoint_eval_metric` attribute of
[config_definition](https://github.com/tensorflow/models/blob/c835649f62994af402c86caab202449a6c8e2f49/official/core/config_definitions.py#L274)
can be used for exporting the best checkpoint, specifying the evaluation metric
the trainer should monitor. Refer to the
[YAML](https://github.com/tensorflow/models/blob/master/official/vision/configs/experiments/semantic_segmentation/deeplabv3plus_resnet101_cityscapes_tfds_tpu.yaml)
file.

--------------------------------------------------------------------------------

## Glossary

Acronym | Meaning
------- | --------------------------
TFM     | Tensorflow Models
FAQs    | Frequently Asked Questions
YAQ     | Yet Another Question
TF      | TensorFlow
