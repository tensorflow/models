# Copyright 2019 The TensorFlow Authors All Rights Reserved.
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
# ==============================================================================
"""Module to construct object detector function."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def MakeDetector(sess, model_dir, import_scope=None):
  """Creates a function to detect objects in an image.

  Args:
    sess: TensorFlow session to use.
    model_dir: Directory where SavedModel is located.
    import_scope: Optional scope to use for model.

  Returns:
    Function that receives an image and returns detection results.
  """
  tf.saved_model.loader.load(
      sess, [tf.saved_model.tag_constants.SERVING],
      model_dir,
      import_scope=import_scope)
  import_scope_prefix = import_scope + '/' if import_scope is not None else ''
  input_images = sess.graph.get_tensor_by_name('%sinput_images:0' %
                                               import_scope_prefix)
  boxes = sess.graph.get_tensor_by_name('%sdetection_boxes:0' %
                                        import_scope_prefix)
  scores = sess.graph.get_tensor_by_name('%sdetection_scores:0' %
                                         import_scope_prefix)
  class_indices = sess.graph.get_tensor_by_name('%sdetection_classes:0' %
                                                import_scope_prefix)

  def DetectorFn(images):
    """Receives an image and returns detected boxes.

    Args:
      images: Uint8 array with shape (batch, height, width 3) containing a batch
        of RGB images.

    Returns:
      Tuple (boxes, scores, class_indices).
    """
    return sess.run([boxes, scores, class_indices],
                    feed_dict={input_images: images})

  return DetectorFn
