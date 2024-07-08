# Customize Training Launcher

[TOC]
## Customize trainer

### Motivation

Customizing the Trainer can be useful for several reasons. One reason may be to
replace or modify the behavior of the existing
[base trainer](https://github.com/tensorflow/models/blob/master/official/core/base_trainer.py#L15)
in TFM. This can be especially useful when a specific use case or problem
requires a unique approach that cannot be easily handled by the pre-existing
training functions. Therefore, customizing the Trainer can give you more
flexibility and control over the training process and help you achieve better
performance on your specific task.

### Instructions

To create a customize trainer , user need to follow the below steps:

#### Create a subclass

To customize a Trainer in TFM, users can subclass the Model Garden
[base Trainer](https://github.com/tensorflow/models/blob/master/official/core/base_trainer.py#L64)
and override the methods that you want to modify. For example, you can override
the `train_loop_end` and `eval_end` methods to process training results and
evaluation results respectively, or you can override the `train_step` method to
define a custom training loop and `eval_step` method to define a custom
validation loop. Additionally, you can override `next_train_inputs` and
`next_eval_inputs` to fetch the next inputs for the model during training and
evaluation.

Here is an example of customizing the Trainer by subclassing the [base Trainer](https://github.com/tensorflow/models/blob/master/official/core/base_trainer.py#L137):

```python
class CustomTrainer(base_trainer.Trainer):
 def __init__(
     self,
     config: ExperimentConfig,
     task: base_task.Task,
     model: tf.keras.Model,
     optimizer: tf.optimizers.Optimizer,
     train_dataset: Optional[Union[tf.data.Dataset,
           tf.distribute.DistributedDataset]] = None,………):

   super().__init__(
       config=config,
       task=task,
       model=model,
       optimizer=optimizer,
       train_dataset=train_dataset,
       ………

 def train_step(self, iterator):
   def step_fn(inputs):
     if self.config.runtime.enable_xla and 
                           (self.config.runtime.num_gpus > 0):
       task_train_step = tf.function(self.task.train_step, 
                                jit_compile=True)
     else:
       task_train_step = self.task.train_step
     logs = task_train_step(………)
     ………

 def eval_step(self, iterator):
   def step_fn(inputs):
     logs = self.task.validation_step(………)
     ………
     return logs
   inputs, passthrough_logs = self.next_eval_inputs(iterator)
   ………

   logs = tf.nest.map_structure(………)
   return passthrough_logs | logs

 def train_loop_end(self):
   self.join()
   logs = {}
   for metric in self.train_metrics + [self.train_loss]:
     logs[metric.name] = metric.result()
     metric.reset_states()
     if hasattr(self.optimizer, 'iterations'):
       logs['learning_rate'] = self.optimizer.learning_rate(
           self.optimizer.iterations)
     ………
   ………

   logs['opimizer_iterations'] = self.optimizer.iterations
   logs['model_global_step'] = self.model._global_step 
   return logs

 def eval_end(self, aggregated_logs=None):
   self.join()
   logs = {}
   for metric in self.validation_metrics:
     logs[metric.name] = metric.result()
   if self.validation_loss.count.numpy() != 0:
     logs[self.validation_loss.name] = self.validation_loss.result()
     ………

   if aggregated_logs:
     metrics = self.task.reduce_aggregated_logs(
         aggregated_logs, global_step=self.global_step)
     logs.update(metrics)

   if self._checkpoint_exporter:
     self._checkpoint_exporter.maybe_export_checkpoint(
         self.checkpoint, logs, self.global_step.numpy())
     ………

   return logs
```
## Customize launch script / Training driver

### Motivation

[Train.py](https://github.com/tensorflow/models/blob/master/official/vision/train.py)
is a script that is used to start model training in TFM. However, in some cases,
you may want to customize the train.py script to suit your specific
requirements. Custom
[train.py](https://github.com/tensorflow/models/blob/master/official/vision/train.py)
can be useful in a variety of situations, particularly in scenarios where
[standard Trainer](https://github.com/tensorflow/models/blob/master/official/core/base_trainer.py#L137)
do not address specific functionality. In such cases, users may need to create a
custom trainer and integrate it into the custom launch script.

Therefore, users might want to customize a training driver to incorporate
specific features or functionalities that are not currently available. Below are
some essential steps to customize a training driver.

### Instructions

To develop your own training driver, you can start by branching out from
standard TFM
[training driver](https://github.com/tensorflow/models/blob/master/official/vision/train.py),
users need to follow the below steps:

**Import the registry**

Ensure that you import the registry. All custom registries and necessary imports
for registration are imported from
[registry_imports.py](https://github.com/tensorflow/models/blob/master/official/vision/registry_imports.py).
Custom models, tasks, configs, etc need to be imported to the registry, so they
can be picked up by the training driver. They can be included in this file so
you do not need to handle each file separately.

If necessary, you can create your own custom registry, refer custom
[registry_imports.py](https://github.com/tensorflow/models/blob/master/official/vision/registry_imports.py)
file here. Please consult the provided syntax as a reference.

```python
  from official import vision
  import registry_imports # pylint: disable=unused-import
```

**Define main method**

The main method in train.py is the entry point of the script that is responsible
for orchestrating the training process. It is the starting point from where the
procedure is executed.
​[​run_experiment](https://github.com/tensorflow/models/blob/master/official/core/train_lib.py#L309)
method is called within the main method and it runs train and eval configured by
the experiment params. It returns a 2-tuple of (model, eval_logs),
`tf.keras.Model` instance and returns eval metrics logs when `run_post_eval` is
set to True, otherwise, returns {}.
[Save_gin_config](https://github.com/tensorflow/models/blob/master/official/core/train_utils.py#L405)
method Serializes and saves the experiment config.

Additional methods other than the main method can be added to the custom
    training driver class to provide additional functionality. Functionalities
    such as loading and saving the model weights, logging training progress to a
    file, sending training progress notifications to certain channels etc. These
    methods can be called from the main method.

Here is an example of how to create a custom launch script :

```python
def main(_):
   ………
  if params.runtime.mixed_precision_dtype:
     performance.set_mixed_precision_policy(
            params.runtime.mixed_precision_dtype)
        distribution_strategy =
        distribute_utils.get_distribution_strategy(
        dist_strategy=params.runtime.distribution_strategy,
        all_reduce_alg=params.runtime.all_reduce_alg,
        num_gpus=params.runtime.num_gpus,
        tpu_address=params.runtime.tpu)

  with distribution_strategy.scope():
     task = task_factory.get_task(params.task,
                            logging_dir=model_dir)
   ………
  train_lib.run_experiment(
  distribution_strategy=distribution_strategy,
  task=task,
  mode=FLAGS.mode,
  params=params,
  model_dir=model_dir)
  train_utils.save_gin_config(FLAGS.mode, model_dir)
   ………

if __name__ == '__main__':

tfm_flags.define_flags()
flags.mark_flags_as_required(['experiment', 'mode','model_dir'])
app.run(main)

```