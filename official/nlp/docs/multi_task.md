# Multi-task library

## Overview

The multi-task library offers lite-weight interfaces and common components to support multi-task
training and evaluation. It makes no assumption on task types and specific model
structure details, instead, it is designed to be a scaffold that effectively
compose single tasks together. Common training scheduling are implemented in the
default module and it saves possibility for further extension on customized use
cases.

The multi-task library support:

-   *joint* training: individual tasks perform forward passes to get a joint
    loss and one backward pass happens.
-   *alternative* training: individual tasks perform independent forward and
    backward pass. The mixture of tasks is controlled by sampling different
    tasks for train steps.

## Library components

### Interfaces

*   [multitask.py](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/multitask.py#L15)
    serves as a stakeholder of multiple
    [`Task`](https://github.com/tensorflow/models/blob/master/official/core/base_task.py#L34)
    instances as well as holding information about multi-task scheduling, such
    as task weight.

*   [base_model.py](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/base_model.py)
    offers access to each single task's forward computation, where each task is
    represented as a `tf.keras.Model` instance. Parameter sharing between tasks
    is left to concrete implementation.

### Common components

*   [base_trainer.py](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/base_trainer.py)
    provides an abstraction to optimize a multi-task model that involves with
    heterogeneous datasets. By default it conducts joint backward step. Task can
    be balanced through setting different task weight on corresponding task
    loss.

*   [interleaving_trainer.py](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/interleaving_trainer.py)
    derives from base trainer and hence shares its housekeeping logic such as
    loss, metric aggregation and reporting. Unlike the base trainer which
    conducts joint backward step, interleaving trainer alternates between tasks
    and effectively mixes single task training step on heterogeneous data sets.
    Task sampling with respect to a probabilistic distribution will be supported
    to facilitate task balancing.

*   [evaluator.py](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/evaluator.py)
    conducts a combination of evaluation of each single task. It simply loops
    through specified tasks and conducts evaluation with corresponding data
    sets.

*   [train_lib.py](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/train_lib.py)
    puts together model, tasks then trainer and triggers training evaluation
    execution.

*   [configs.py](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/configs.py)
    provides a top level view on the entire system. Configuration objects are
    mimicked or composed from corresponding single task components to reuse
    whenever possible and maintain consistency. For example,
    [`TaskRoutine`](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/configs.py#L25)
    effectively reuses
    [`Task`](https://github.com/tensorflow/models/blob/master/official/core/base_task.py#L34);
    and
    [`MultiTaskConfig`](https://github.com/tensorflow/models/blob/master/official/modeling/multitask/configs.py#L34)
    serves as a similar role of
    [`TaskConfig`](https://github.com/tensorflow/models/blob/master/official/core/config_definitions.py#L211)

### Notes on single task composability

The library is designed to be able to put together multi-task model by composing
single task implementations. This is reflected in many aspects:

*   Base model interface allows single task's `tf.keras.Model` implementation to
    be reused, given the shared parts in a potential multi-task case are passed
    in through constructor. A good example of this is
    [`BertClassifier`](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/bert_classifier.py#L24)
    and
    [`BertSpanLabeler`](https://github.com/tensorflow/models/blob/master/official/nlp/modeling/models/bert_span_labeler.py),
    where the backbone network is initialized out of classifier object. Hence a
    multi-task model that conducts both classification + sequence labeling using
    a shared backbone encoder could be easily created from existing code.

*   Multi-task interface holds a set of Task objects, hence completely reuse the
    input functions, loss functions, metrics with corresponding aggregation and
    reduction logic. **Note, under multi-task training situation, the
    [`build_model()`](https://github.com/tensorflow/models/blob/master/official/core/base_task.py#L144)
    are not used**, given partially shared structure cannot be specified with
    only one single task.

*   Interleaving trainer works on top of each single task's
    [`train_step()`](https://github.com/tensorflow/models/blob/master/official/core/base_task.py#L223).
    This hides the optimization details from each single task and focuses on
    optimization scheduling and task balancing.