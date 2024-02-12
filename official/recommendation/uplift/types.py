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

"""Defines types used by the keras uplift modeling library."""

import tensorflow as tf, tf_keras

TensorType = tf.Tensor | tf.SparseTensor | tf.RaggedTensor

ListOfTensors = list[TensorType]
TupleOfTensors = tuple[TensorType, ...]
DictOfTensors = dict[str, TensorType]

CollectionOfTensors = ListOfTensors | TupleOfTensors | DictOfTensors


class TwoTowerNetworkOutputs(tf.experimental.ExtensionType):
  """Tensors computed by a `BaseTwoTowerUpliftNetwork` layer.

  Attributes:
    shared_embedding: embedding computed by the backbone layer and used as a
      shared representation for the control and treatment towers.
    control_logits: logits for the control group. Its shape and dtype must be
      the same as the treatment logits.
    treatment_logits: logits for the treatment group. Its shape and dtype must
      be the same as the control logits.
  """

  __name__ = "TwoTowerNetworkOutputs"

  shared_embedding: tf.Tensor
  control_logits: tf.Tensor
  treatment_logits: tf.Tensor

  # TODO(b/281776818): Override __validate__ to assert control and treatment
  # logits are of the same dtype and shape. Also add validation tests.

  # The `model.compile()` API casts and expands labels plus sample weights to
  # match the shape and dtype of the model outputs. By setting the dtype and
  # shape to that of the control and treatment logits the labels and sample
  # weights get casted to the same dtype and shape as that of the logits.
  dtype = property(lambda self: self.control_logits.dtype)
  shape = property(lambda self: self.control_logits.shape)

  class Spec:
    """Tensor spec.

    Note that this ExtensionType does not have a well defined shape since its
    intended use is the same as that of a dataclass. A noteworthy case of when
    the spec's shape attribute is needed is when a `KerasTensor` is initialized
    during the construction of a functional Keras model, which expects all
    tensors to have a spec with a shape attribute.
    """

    shape = property(lambda self: tf.TensorShape(None))


class TwoTowerPredictionOutputs(TwoTowerNetworkOutputs):
  """Inference tensors computed by a `TwoTowerUpliftNetwork` layer.

  Attributes:
    control_predictions: predictions for the control group. Its shape and dtype
      must be the same as the treatment predictions.
    treatment_predictions: predictions for the treatment group. Its shape and
      dtype must be the same as the control predictions.
    uplift: difference between the treatment and control predictions.
  """

  __name__ = "TwoTowerPredictionOutputs"

  control_predictions: tf.Tensor
  treatment_predictions: tf.Tensor
  uplift: tf.Tensor

  # TODO(b/281776818): Override __validate__ to assert control and treatment
  # predictions are of the same dtype and shape as the control and treatment
  # logits. Also add validation tests.

  class Spec:
    shape = property(lambda self: tf.TensorShape(None))


class TwoTowerTrainingOutputs(TwoTowerPredictionOutputs):
  """Training tensors computed by a `TwoTowerUpliftNetwork` layer.

  Attributes:
    true_logits: logits for either the control or treatment group, depending on
      the corresponding value in the `is_treatment` tensor. It will contain
      treatment group logits for the `is_treatment == 1` entries and control
      group logits otherwise.
    true_predictions: predictions for either the control or treatment group,
      depending on the corresponding value in the `is_treatment` tensor. It will
      contain treatment group predictions for the `is_treatment == 1` entries
      and control group predictions otherwise.
    is_treatment: a boolean `tf.Tensor` indicating if the example belongs to the
      treatment group (True) or control group (False).
  """

  __name__ = "TwoTowerTrainingOutputs"

  true_logits: tf.Tensor
  true_predictions: tf.Tensor
  is_treatment: tf.Tensor

  # TODO(b/281776818): Override __validate__ to assert that the true logits is
  # of the same rank as the control and treatment logits, and that the
  # is_treatment tensor is a boolean tensor. Also add validation tests.

  class Spec:
    shape = property(lambda self: tf.TensorShape(None))
