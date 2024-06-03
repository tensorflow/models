# Optimizer and Learning Rate Scheduler

This page describes the
[optimization package](https://github.com/tensorflow/models/tree/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/modeling/optimization)
for Tensorflow Official Models (TFM) which includes optimizers, and learning
rate schedulers.

## Building Optimizer and LR Scheduler

We use an Optimizer factory class to manage optimizer and learning rate
creation. Optimizer factory takes a config as an input, and it has member
functions that are used to build optimizer and learning rate schedule. To create
an optimizer and a LR schedule through OptimizerFactory, you need to do the
following:

1.  Define optimization config, this includes optimizer, and learning rate
    schedule.
2.  Initialize the OptimizerFactory instance using the optimization config.
3.  Build the learning rate, and the optimizer using the class member functions.

The following is an example for creating an SGD optimizer with stepwise LR
scheduler with linear warmup:

```python
params = {'optimizer': { 'type': 'sgd',
                         'sgd': {'momentum': 0.9}},
          'learning_rate': {'type': 'stepwise',
                            'stepwise': {
                                'boundaries': [10000, 20000],
                                'values': [0.1, 0.01, 0.001]}},
          'warmup': {'type': 'linear',
                     'linear': {'warmup_steps': 500,
                                'warmup_learning_rate': 0.01}}}
# Defines optimization config from a dictionary.
opt_config = optimization.OptimizationConfig(params)
# Initializes an optimization factory from optimization config.
opt_factory = optimization.OptimizerFactory(opt_config)
# Builds the desired learning rate scheduling instance.
lr = opt_factory.build_learning_rate()
# Builds the optimizer instance with the desired learning rate schedule.
optimizer = opt_factory.build_optimizer(lr)
```

To initialize an OptimizerFactory, `optimizer` and `learning_rate` fields must
be defined, while `warmup` is an optional field. The field `type` is used to
define the type of each optimization component. The set of available types are
explained in details in the following sections.

In the following sections, we explain how to create different optimizers,
learning rate, and warmup schedulers. We also explain how to add new optimizers,
or learning rate schedulers.

## Optimizers

The list of supported optimizers can be found
[here](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/modeling/optimization/optimizer_factory.py#L31).

```python
OPTIMIZERS_CLS = {
    'sgd': tf.keras.optimizers.SGD,
    'adam': tf.keras.optimizers.Adam,
    'adamw': nlp_optimization.AdamWeightDecay,
    'lamb': tfa_optimizers.LAMB,
    'rmsprop': tf.keras.optimizers.RMSprop
}
```

You can specify the type of optimizer to be one of the above using
[oneof](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/modeling/hyperparams/oneof.py)
config. The available config fields can be found
[here](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/modeling/optimization/configs/optimizer_config.py).

All optimizers support gradient clipping methods: clip by value, clip by norm,
clip by global norm. To specify which method to use, you need to specify the
appropriate field list
[here](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/modeling/optimization/configs/optimizer_config.py#L34).

### Example

We will specify an rmsprop optimizer with discounting factor (rho) of 0.9, and
global norm gradient clipping of 10.0. Below is the config to be used.

```python
params = {'optimizer': { 'type': 'rmsprop',
                         'rmsprop': {'rho': 0.9,
                                     'global_clipnorm': 10.0}}}
```

### Adding a New Optimizer

To add a new optimizer, you need to do the following:

1.  Create a
    [custom](https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer#creating_a_custom_optimizer_2)
    of tf.keras.optimizers.Optimizer.
2.  Add the required config fields under
    [optimization/configs/optimizer_config.py](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/modeling/optimization/configs/optimizer_config.py).
3.  Add the optimizer class to the list of available optimizer classes in
    (optimizer_factor)[https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/modeling/optimization/optimizer_factory.py]

## Learning Rate and Warmup Schedules

Learning rate with an optional warmup can be configured by specifying
`learning_rate`, and `warmup` fields in optimization config. `learning_rate` is
a required field, while `warmup` is an optional one. The list of supported
`learning_rate` and `warmup` schedules can be found
[here](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/modeling/optimization/optimizer_factory.py#L59).

```python
LR_CLS = {
    'stepwise': tf.keras.optimizers.schedules.PiecewiseConstantDecay,
    'polynomial': tf.keras.optimizers.schedules.PolynomialDecay,
    'exponential': tf.keras.optimizers.schedules.ExponentialDecay,
    'cosine': tf.keras.experimental.CosineDecay,
    'power': lr_schedule.DirectPowerDecay,
}

WARMUP_CLS = {
    'linear': lr_schedule.LinearWarmup,
    'polynomial': lr_schedule.PolynomialWarmUp
}
```

In addition, a `constant` learning rate can be specified.

## How Learning Rate Works

Learning rate takes `step` as an input, and it returns the learning rate value.
As the training progresses, usually learning rate value decays. Warmup schedule
is often used to stablize the training. Warmup schedule starts from a low
learning rate value, and it gradually increases until it reaches the initial
value for the regular learning rate decay schedule. We combine `learning_rate`
(lr) with `warmup` (warmup) schedules as follows

*   Steps [0, warmup_steps): `learning_rate = warmup(step)`
*   Steps [warmup_steps, train_steps): `learning_rate = lr(step)`
*   We designed the warmup schedule such that final warmup learning rate is
    inferred from the learning rate schedule (i.e.
    `learning_rate(warmup_steps) = warmup(warmup_steps)`). Note that, warmup
    schedule doesn't delay the regular learning rate decay by warmup_steps,
    instead it replaces it.

Learning rate value is logged every
[summary_interval](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/core/config_definitions.py#L262).
If warmup_steps are less that the `summary_interval`, you won't be able to see
warmup values.

### Example

We want to specify a cosine learning rate decay with decay_steps of 20000, with
a linear warmup schedule for the first 500 steps.

```python
params = {'learning_rate': {'type': 'cosine',
                            'cosine': {'decay_steps': 20000}},
          'warmup': {'type': 'linear',
                     'linear': {'warmup_steps': 500}}}
```

## Customizing Optimizer Inside Task

Optimizer and learning rate are created inside the
[task](https://github.com/tensorflow/models/blob/28d972a0b30b628cbb7f67a090ea564c3eda99ea/official/core/base_task.py#L99).
If different optimizers/learning rate schedulers are needed, they can be defined
by overriding the class method.

## Important Factors To Consider

*   Batch size: Changing batch size usually requires scaling learning rate
    values, and number of training steps. Make sure that you change appropriate
    values as batch size changes.
*   Train steps: Train steps is highly correlated with fields such as
    `decay_steps` for cosine learning rate decay. Changing one without changing
    the other might result in undesired behavior.
