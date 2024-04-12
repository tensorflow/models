# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keras metric for reporting metrics sliced by a feature."""

import copy

import tensorflow as tf, tf_keras


class SlicedMetric(tf_keras.metrics.Metric):
  """A metric sliced by integer, boolean, or string features.

  A metric wrapper that computes a metric for different slices of an arbitrary
  feature. The slicing is specified via a slicing spec, which is a dictionary
  from a slice name to the unique value to be sliced on. For each pair of
  `slice_name`, `slicing_value` passed, the suffix `/slice_name` will be
  appended to the name of the result of the corresponding slice.
  An overall result is also computed without any slicing applied.

  In order for this to work correctly, the given metric must support passing
  `sample_weight` to its `update_state()` method. Additionally, the slicing
  feature must also be passed to `update_state()` method of this class and
  must be of a broadcastable shape to the metric inputs.
  This wrapper creates a deep copy of the metric passed to it for each slice.
  At every call to `update_state()`, the wrapper will call the `update_state()`
  method of the metric for each slice with the `sample_weights` set to zero
  where the slicing feature is not equal to the corresponding slicing value.

  If the given metric returns a tensor, the result of this metric will be a
  dictionary mapping from the sliced metric's name to the result for that slice.
  If the given metric returns a dictionary of tensors, the result of this metric
  will be a flattened dictionary consisting of each of the sliced metrics'
  results for every slice.

  Example usage:

  >>> sliced_metric = SlicedMetric(
  ...     tf_keras.metrics.Accuracy('accuracy'),
  ...     slicing_spec={"control": False, "treatment": True},
  ... )
  >>> sliced_metric.update_state(
  ...     y_true=tf.constant([[0], [1], [0], [1]]),
  ...     y_pred=tf.constant([[1], [0], [1], [1]]),
  ...     slicing_feature=tf.constant([[True], [False], [True], [False]]),
  ... )
  >>> sliced_metric.result()
  {
      "accuracy": 0.25,
      "accuracy/control": 0.5,
      "accuracy/treatment": 0
  }
  """

  def __init__(
      self,
      metric: tf_keras.metrics.Metric,
      slicing_spec: dict[str, str] | dict[str, int],
      slicing_feature_dtype: tf.DType | None = None,
      name: str | None = None,
  ):
    """Initializes the instance.

    Args:
      metric: A `tf_keras.metrics.Metric` instance.
      slicing_spec: A dictionary that maps from string slice names, to one of
        integer, boolean, or string slicing values.
      slicing_feature_dtype: The expected dtype of the slicing feature. The
        values in the slicing spec are casted to this type if passed. If None,
        the dtype of the slicing feature is inferred based on the values in the
        slicing spec.
      name: The name of the wrapper metric. Defaults to `sliced_{metric.name}`.

    Raises:
      A ValueError if `slicing_spec` is empty, contains duplicate slicing
      values, or has slicing values of different types.
    """
    super().__init__(name=name or f"sliced_{metric.name}", dtype=metric.dtype)

    if not slicing_spec:
      raise ValueError("The slicing spec must be a non-empty dictionary.")

    slice_names, slicing_values = zip(*slicing_spec.items())
    if not isinstance(slicing_values[0], (int, bool, str)) or not all(
        isinstance(k, type(slicing_values[0])) for k in slicing_values
    ):
      raise ValueError(
          "All slicing values in the slicing spec must be one of `int`, "
          "`bool`, or `str`, and all values must have the same type. "
          f"Got types: {list(map(type, slicing_values))}."
      )

    if len(slicing_values) > len(set(slicing_values)):
      raise ValueError(
          "The slicing values passed to the slicing spec must be unique. Got "
          f"{slicing_values}."
      )

    # TODO(b/276811843): Look into validating whether `metric` accepts
    # `sample_weights` in its `update_state` method.

    # Instance fully owns a deep copy of the metric.
    self._metric = copy.deepcopy(metric)
    self._slice_names = list(slice_names)
    self._slicing_values = list(slicing_values)
    self._slicing_values_tensors = [
        tf.constant(v, slicing_feature_dtype) for v in slicing_values
    ]
    self._slicing_feature_dtype = self._slicing_values_tensors[0].dtype
    self._sliced_metrics = [copy.deepcopy(metric) for _ in self._slicing_values]

  def update_state(
      self,
      *args: tf.Tensor,
      sample_weight: tf.Tensor | None = None,
      slicing_feature: tf.Tensor,
      **kwargs,
  ):
    """Updates the state of the metrics for each slice.

    Args:
      *args: A variable amount of `tf.Tensor` instances that will be passed to
        the `update_state` method of each metric.
      sample_weight: An optional `tf.Tensor` used to weight the sample. Its
        dimensions must be broadcastable to the shape(s) of *args.
      slicing_feature: A `tf.Tensor` consisting of the feature to be sliced on.
        Its dimensions must be broadcastable to the shape(s) of *args.
      **kwargs: Keyword arguments that will be passed to the `update_state`
        method of each metric.
    """

    if slicing_feature.dtype != self._slicing_feature_dtype:
      raise ValueError(
          "The `slicing_feature` and slicing values in `slicing_spec` must "
          "have the same type. Got types: "
          f"{(slicing_feature.dtype, self._slicing_feature_dtype)}."
      )

    if sample_weight is not None:
      for _ in range(len(slicing_feature.shape) - len(sample_weight.shape)):
        sample_weight = tf.expand_dims(sample_weight, axis=-1)

      for _ in range(len(sample_weight.shape) - len(slicing_feature.shape)):
        slicing_feature = tf.expand_dims(slicing_feature, axis=-1)

    self._metric.update_state(*args, sample_weight=sample_weight, **kwargs)
    for slicing_val, metric in zip(
        self._slicing_values_tensors, self._sliced_metrics
    ):
      slice_mask = tf.cast(slicing_feature == slicing_val, dtype=tf.float32)
      if sample_weight is not None:
        weight = slice_mask * tf.cast(sample_weight, dtype=tf.float32)
      else:
        weight = slice_mask
      metric.update_state(*args, sample_weight=weight, **kwargs)

  def result(self) -> dict[str, tf.Tensor]:
    """Aggregates all the metrics' results into a flattened dictionary."""
    metric_name = self._metric.name
    metric_result = self._metric.result()
    slice_results = [metric.result() for metric in self._sliced_metrics]

    if isinstance(metric_result, tf.Tensor):
      results = {metric_name: metric_result}
      slice_names = (f"{metric_name}/{name}" for name in self._slice_names)
      results.update(zip(slice_names, slice_results))
      return results

    if isinstance(metric_result, dict) and all(
        isinstance(result, tf.Tensor) for result in metric_result.values()
    ):
      results = {**metric_result}
      for slice_name, slice_result in zip(self._slice_names, slice_results):
        result_names, result_values = zip(*slice_result.items())
        slice_names = [f"{name}/{slice_name}" for name in result_names]
        results.update(zip(slice_names, result_values))
      return results

    raise ValueError(
        "The output of the given metric must either be a `tf.Tensor` or "
        "a `dict[str, tf.Tensor]`, but got unsupported output: "
        f"{metric_result}."
    )

  def reset_state(self):
    self._metric.reset_state()
    for metric in self._sliced_metrics:
      metric.reset_state()

  def get_config(self):
    return {
        "name": self.name,
        "metric": tf_keras.metrics.serialize(self._metric),
        "slicing_spec": dict(zip(self._slice_names, self._slicing_values)),
        "slicing_feature_dtype": self._slicing_feature_dtype.name,
    }

  @classmethod
  def from_config(cls, config):
    config["metric"] = tf_keras.metrics.deserialize(config["metric"])
    config["slicing_feature_dtype"] = tf.as_dtype(
        config["slicing_feature_dtype"]
    )
    return cls(**config)
