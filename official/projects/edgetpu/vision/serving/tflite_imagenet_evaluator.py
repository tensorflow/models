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

"""Evaluates image classification accuracy using TFLite Interpreter."""

import dataclasses
import multiprocessing.pool as mp
from typing import Tuple
from absl import logging
import numpy as np
import tensorflow as tf, tf_keras

# pylint: disable=g-direct-tensorflow-import
from tensorflow.lite.python import interpreter as tfl_interpreter
# pylint: enable=g-direct-tensorflow-import


@dataclasses.dataclass
class EvaluationInput():
  """Contains image and its label as evaluation input."""
  image: tf.Tensor
  label: tf.Tensor


class AccuracyEvaluator():
  """Evaluates image classification accuracy using TFLite Interpreter.

  Attributes:
    model_content: The contents of a TFLite model.
    num_threads: Number of threads used to evaluate images.
    thread_batch_size: Batch size assigned to each thread.
    image_size: Width/Height of the images.
    num_classes: Number of classes predicted by the model.
    resize_method: Resize method to use during image preprocessing.
  """

  def __init__(self,
               model_content: bytes,
               dataset: tf.data.Dataset,
               num_threads: int = 16):
    self._model_content: bytes = model_content
    self._dataset = dataset
    self._num_threads: int = num_threads

  def evaluate_single_image(self, eval_input: EvaluationInput) -> bool:
    """Evaluates a given single input.

    Args:
      eval_input: EvaluationInput holding image and label.

    Returns:
      Whether the estimation is correct.
    """
    interpreter = tfl_interpreter.Interpreter(
        model_content=self._model_content, num_threads=1
    )
    interpreter.allocate_tensors()
    # Get input and output tensors and quantization details.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    image_tensor = interpreter.tensor(input_details[0]['index'])
    logits_tensor = interpreter.tensor(output_details[0]['index'])

    # Handle quantization.
    scale = 1.0
    zero_point = 0.0
    input_dtype = tf.as_dtype(input_details[0]['dtype'])
    if input_dtype.is_quantized or input_dtype.is_integer:
      input_quantization = input_details[0]['quantization']
      scale = input_quantization[0]
      zero_point = input_quantization[1]
    image_tensor()[0, :] = (eval_input.image.numpy() / scale) + zero_point

    interpreter.invoke()
    return eval_input.label.numpy() == np.argmax(logits_tensor()[0])

  def evaluate_all(self) -> Tuple[int, int]:
    """Evaluates all of images in the default dataset.

    Returns:
      Total number of evaluations and correct predictions as tuple of ints.
    """
    num_evals = 0
    num_corrects = 0
    for image_batch, label_batch in self._dataset:
      inputs = [
          EvaluationInput(image, label)
          for image, label in zip(image_batch, label_batch)
      ]
      pool = mp.ThreadPool(self._num_threads)
      results = pool.map(self.evaluate_single_image, inputs)
      pool.close()
      pool.join()
      num_evals += len(results)
      num_corrects += results.count(True)
      accuracy = 100.0 * num_corrects / num_evals if num_evals > 0 else 0
      logging.info('Evaluated: %d, Correct: %d, Accuracy: %f', num_evals,
                   num_corrects, accuracy)
    return (num_evals, num_corrects)
