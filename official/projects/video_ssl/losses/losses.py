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

"""Define losses."""

# Import libraries
import tensorflow as tf, tf_keras
from tensorflow.compiler.tf2xla.python import xla


def contrastive_loss(hidden,
                     num_replicas,
                     normalize_hidden,
                     temperature,
                     model,
                     weight_decay):
  """Computes contrastive loss.

  Args:
    hidden: embedding of video clips after projection head.
    num_replicas: number of distributed replicas.
    normalize_hidden: whether or not to l2 normalize the hidden vector.
    temperature: temperature in the InfoNCE contrastive loss.
    model: keras model for calculating weight decay.
    weight_decay: weight decay parameter.

  Returns:
    A loss scalar.
    The logits for contrastive prediction task.
    The labels for contrastive prediction task.
  """
  large_num = 1e9

  hidden1, hidden2 = tf.split(hidden, num_or_size_splits=2, axis=0)
  if normalize_hidden:
    hidden1 = tf.math.l2_normalize(hidden1, -1)
    hidden2 = tf.math.l2_normalize(hidden2, -1)
  batch_size = tf.shape(hidden1)[0]

  if num_replicas == 1:
    # This is the local version
    hidden1_large = hidden1
    hidden2_large = hidden2
    labels = tf.one_hot(tf.range(batch_size), batch_size * 2)
    masks = tf.one_hot(tf.range(batch_size), batch_size)

  else:
    # This is the cross-tpu version.
    hidden1_large = tpu_cross_replica_concat(hidden1, num_replicas)
    hidden2_large = tpu_cross_replica_concat(hidden2, num_replicas)
    enlarged_batch_size = tf.shape(hidden1_large)[0]
    replica_id = tf.cast(tf.cast(xla.replica_id(), tf.uint32), tf.int32)
    labels_idx = tf.range(batch_size) + replica_id * batch_size
    labels = tf.one_hot(labels_idx, enlarged_batch_size * 2)
    masks = tf.one_hot(labels_idx, enlarged_batch_size)

  logits_aa = tf.matmul(hidden1, hidden1_large, transpose_b=True) / temperature
  logits_aa = logits_aa - tf.cast(masks, logits_aa.dtype) * large_num
  logits_bb = tf.matmul(hidden2, hidden2_large, transpose_b=True) / temperature
  logits_bb = logits_bb - tf.cast(masks, logits_bb.dtype) * large_num
  logits_ab = tf.matmul(hidden1, hidden2_large, transpose_b=True) / temperature
  logits_ba = tf.matmul(hidden2, hidden1_large, transpose_b=True) / temperature

  loss_a = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      labels, tf.concat([logits_ab, logits_aa], 1)))
  loss_b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
      labels, tf.concat([logits_ba, logits_bb], 1)))
  loss = loss_a + loss_b

  l2_loss = weight_decay * tf.add_n([
      tf.nn.l2_loss(v)
      for v in model.trainable_variables
      if 'kernel' in v.name
    ])

  total_loss = loss +  tf.cast(l2_loss, loss.dtype)

  contrast_prob = tf.nn.softmax(logits_ab)
  contrast_entropy = - tf.reduce_mean(
      tf.reduce_sum(contrast_prob * tf.math.log(contrast_prob + 1e-8), -1))

  contrast_acc = tf.equal(tf.argmax(labels, 1), tf.argmax(logits_ab, axis=1))
  contrast_acc = tf.reduce_mean(tf.cast(contrast_acc, tf.float32))

  return {
      'total_loss': total_loss,
      'contrastive_loss': loss,
      'reg_loss': l2_loss,
      'contrast_acc': contrast_acc,
      'contrast_entropy': contrast_entropy,
  }


def tpu_cross_replica_concat(tensor, num_replicas):
  """Reduce a concatenation of the `tensor` across TPU cores.

  Args:
    tensor: tensor to concatenate.
    num_replicas: number of TPU device replicas.

  Returns:
    Tensor of the same rank as `tensor` with first dimension `num_replicas`
    times larger.
  """
  with tf.name_scope('tpu_cross_replica_concat'):
    # This creates a tensor that is like the input tensor but has an added
    # replica dimension as the outermost dimension. On each replica it will
    # contain the local values and zeros for all other values that need to be
    # fetched from other replicas.
    ext_tensor = tf.scatter_nd(
        indices=[[xla.replica_id()]],
        updates=[tensor],
        shape=[num_replicas] + tensor.shape.as_list())

    # As every value is only present on one replica and 0 in all others, adding
    # them all together will result in the full tensor on all replicas.
    replica_context = tf.distribute.get_replica_context()
    ext_tensor = replica_context.all_reduce(tf.distribute.ReduceOp.SUM,
                                            ext_tensor)

    # Flatten the replica dimension.
    # The first dimension size will be: tensor.shape[0] * num_replicas
    # Using [-1] trick to support also scalar input.
    return tf.reshape(ext_tensor, [-1] + ext_tensor.shape.as_list()[2:])
