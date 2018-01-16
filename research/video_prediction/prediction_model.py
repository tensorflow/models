# Copyright 2016 The TensorFlow Authors All Rights Reserved.
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

"""Model architecture for predictive model, including CDNA, DNA, and STP."""

import numpy as np
import tensorflow as tf

import tensorflow.contrib.slim as slim
from tensorflow.python.platform import flags
from tensorflow.contrib.layers.python import layers as tf_layers
from lstm_ops import basic_conv_lstm_cell

FLAGS = flags.FLAGS

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12

# kernel size for DNA and CDNA.
DNA_KERN_SIZE = 5


def kl_divergence(mu, log_sigma):
  """KL divergence of diagonal gaussian N(mu,exp(log_sigma)) and N(0,1).

  Args:
    mu: mu parameter of the distribution.
    log_sigma: log(sigma) parameter of the distribution.
  Returns:
    the KL loss.
  """

  return -.5 * tf.reduce_sum(1. + log_sigma - tf.square(mu) - tf.exp(log_sigma),
                             axis=1)


def construct_latent_tower(images):
  """Builds convolutional latent tower for stochastic model.

  At training time this tower generates a latent distribution (mean and std)
  conditioned on the entire video. This latent variable will be fed to the
  main tower as an extra variable to be used for future frames prediction.
  At inference time, the tower is disabled and only returns latents sampled
  from N(0,1).
  If the multi_latent flag is on, a different latent for every timestep would
  be generated.

  Args:
    images: tensor of ground truth image sequences
  Returns:
    latent_mean: predicted latent mean
    latent_std: predicted latent standard deviation
    latent_loss: loss of the latent twoer
    samples: random samples sampled from standard guassian
  """

  with slim.arg_scope([slim.conv2d], reuse=False):
    stacked_images = tf.concat(images, 3)

    latent_enc1 = slim.conv2d(
        stacked_images,
        32, [3, 3],
        stride=2,
        scope='latent_conv1',
        normalizer_fn=tf_layers.layer_norm,
        normalizer_params={'scope': 'latent_norm1'})

    latent_enc2 = slim.conv2d(
        latent_enc1,
        64, [3, 3],
        stride=2,
        scope='latent_conv2',
        normalizer_fn=tf_layers.layer_norm,
        normalizer_params={'scope': 'latent_norm2'})

    latent_enc3 = slim.conv2d(
        latent_enc2,
        64, [3, 3],
        stride=1,
        scope='latent_conv3',
        normalizer_fn=tf_layers.layer_norm,
        normalizer_params={'scope': 'latent_norm3'})

    latent_mean = slim.conv2d(
        latent_enc3,
        FLAGS.latent_channels, [3, 3],
        stride=2,
        activation_fn=None,
        scope='latent_mean',
        normalizer_fn=tf_layers.layer_norm,
        normalizer_params={'scope': 'latent_norm_mean'})

    latent_std = slim.conv2d(
        latent_enc3,
        FLAGS.latent_channels, [3, 3],
        stride=2,
        scope='latent_std',
        normalizer_fn=tf_layers.layer_norm,
        normalizer_params={'scope': 'latent_std_norm'})

    latent_std += FLAGS.latent_std_min

    divergence = kl_divergence(latent_mean, latent_std)
    latent_loss = tf.reduce_mean(divergence)

  if FLAGS.multi_latent:
    # timestep x batch_size x latent_size
    samples = tf.random_normal(
        [FLAGS.sequence_length-1] + latent_mean.shape, 0, 1,
        dtype=tf.float32)
  else:
    # batch_size x latent_size
    samples = tf.random_normal(latent_mean.shape, 0, 1, dtype=tf.float32)

  if FLAGS.inference_time:
    # No latent tower at inference time, just standard gaussian.
    return None, None, None, samples
  else:
    return latent_mean, latent_std, latent_loss, samples


def construct_model(images,
                    actions=None,
                    states=None,
                    iter_num=-1.0,
                    k=-1,
                    use_state=True,
                    num_masks=10,
                    stp=False,
                    cdna=True,
                    dna=False,
                    context_frames=2):
  """Build convolutional lstm video predictor using STP, CDNA, or DNA.

  Args:
    images: tensor of ground truth image sequences
    actions: tensor of action sequences
    states: tensor of ground truth state sequences
    iter_num: tensor of the current training iteration (for sched. sampling)
    k: constant used for scheduled sampling. -1 to feed in own prediction.
    use_state: True to include state and action in prediction
    num_masks: the number of different pixel motion predictions (and
               the number of masks for each of those predictions)
    stp: True to use Spatial Transformer Predictor (STP)
    cdna: True to use Convoluational Dynamic Neural Advection (CDNA)
    dna: True to use Dynamic Neural Advection (DNA)
    context_frames: number of ground truth frames to pass in before
                    feeding in own predictions
  Returns:
    gen_images: predicted future image frames
    gen_states: predicted future states

  Raises:
    ValueError: if more than one network option specified or more than 1 mask
    specified for DNA model.
  """
  # Each image is being used twice, in latent tower and main tower.
  # This is to make sure we are using the *same* image for both, ...
  # ... given how TF queues work. 
  images = [tf.identity(image) for image in images]

  if stp + cdna + dna != 1:
    raise ValueError('More than one, or no network option specified.')
  batch_size, img_height, img_width, color_channels = images[0].get_shape()[0:4]
  lstm_func = basic_conv_lstm_cell

  # Generated robot states and images.
  gen_states, gen_images = [], []
  current_state = states[0]

  if k == -1:
    feedself = True
  else:
    # Scheduled sampling:
    # Calculate number of ground-truth frames to pass in.
    num_ground_truth = tf.to_int32(
        tf.round(tf.to_float(batch_size) * (k / (k + tf.exp(iter_num / k)))))
    feedself = False

  # LSTM state sizes and states.
  lstm_size = np.int32(np.array([32, 32, 64, 64, 128, 64, 32]))
  lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
  lstm_state5, lstm_state6, lstm_state7 = None, None, None

  # Latent tower
  latent_loss = 0.0
  if FLAGS.stochastic_model:
    latent_tower_outputs = construct_latent_tower(images)
    latent_mean, latent_std, latent_loss, samples = latent_tower_outputs

  # Main tower    
  for image, action in zip(images[:-1], actions[:-1]):
    # Reuse variables after the first timestep.
    reuse = bool(gen_images)

    done_warm_start = len(gen_images) > context_frames - 1
    with slim.arg_scope(
        [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
         tf_layers.layer_norm, slim.layers.conv2d_transpose],
        reuse=reuse):

      if feedself and done_warm_start:
        # Feed in generated image.
        prev_image = gen_images[-1]
      elif done_warm_start:
        # Scheduled sampling
        prev_image = scheduled_sample(image, gen_images[-1], batch_size,
                                      num_ground_truth)
      else:
        # Always feed in ground_truth
        prev_image = image

      # Predicted state is always fed back in
      state_action = tf.concat(axis=1, values=[action, current_state])

      enc0 = slim.layers.conv2d(
          prev_image,
          32, [5, 5],
          stride=2,
          scope='scale1_conv1',
          normalizer_fn=tf_layers.layer_norm,
          normalizer_params={'scope': 'layer_norm1'})

      hidden1, lstm_state1 = lstm_func(
          enc0, lstm_state1, lstm_size[0], scope='state1')
      hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')
      hidden2, lstm_state2 = lstm_func(
          hidden1, lstm_state2, lstm_size[1], scope='state2')
      hidden2 = tf_layers.layer_norm(hidden2, scope='layer_norm3')
      enc1 = slim.layers.conv2d(
          hidden2, hidden2.get_shape()[3], [3, 3], stride=2, scope='conv2')

      hidden3, lstm_state3 = lstm_func(
          enc1, lstm_state3, lstm_size[2], scope='state3')
      hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')
      hidden4, lstm_state4 = lstm_func(
          hidden3, lstm_state4, lstm_size[3], scope='state4')
      hidden4 = tf_layers.layer_norm(hidden4, scope='layer_norm5')
      enc2 = slim.layers.conv2d(
          hidden4, hidden4.get_shape()[3], [3, 3], stride=2, scope='conv3')

      # Pass in state and action.
      smear = tf.reshape(
          state_action,
          [int(batch_size), 1, 1, int(state_action.get_shape()[1])])
      smear = tf.tile(
          smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])
      if use_state:
        enc2 = tf.concat(axis=3, values=[enc2, smear])
        
      # Setup latent
      if FLAGS.stochastic_model:
        latent = samples
        if FLAGS.multi_latent:
          latent = samples[timestep]
        if not FLAGS.inference_time:
          latent = tf.cond(iter_num < FLAGS.num_iterations_1st_stage,
                           lambda: tf.identity(latent),
                           lambda: latent_mean + tf.exp(latent_std / 2.0) * latent)
        with tf.control_dependencies([latent]):
          enc2 = tf.concat([enc2, latent], 3)
        
      enc3 = slim.layers.conv2d(
          enc2, hidden4.get_shape()[3], [1, 1], stride=1, scope='conv4')

      hidden5, lstm_state5 = lstm_func(
          enc3, lstm_state5, lstm_size[4], scope='state5')  # last 8x8
      hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
      enc4 = slim.layers.conv2d_transpose(
          hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')

      hidden6, lstm_state6 = lstm_func(
          enc4, lstm_state6, lstm_size[5], scope='state6')  # 16x16
      hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')
      # Skip connection.
      hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16

      enc5 = slim.layers.conv2d_transpose(
          hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
      hidden7, lstm_state7 = lstm_func(
          enc5, lstm_state7, lstm_size[6], scope='state7')  # 32x32
      hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

      # Skip connection.
      hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

      enc6 = slim.layers.conv2d_transpose(
          hidden7,
          hidden7.get_shape()[3], 3, stride=2, scope='convt3', activation_fn=None,
          normalizer_fn=tf_layers.layer_norm,
          normalizer_params={'scope': 'layer_norm9'})

      if dna:
        # Using largest hidden state for predicting untied conv kernels.
        enc7 = slim.layers.conv2d_transpose(
            enc6, DNA_KERN_SIZE**2, 1, stride=1, scope='convt4', activation_fn=None)
      else:
        # Using largest hidden state for predicting a new image layer.
        enc7 = slim.layers.conv2d_transpose(
            enc6, color_channels, 1, stride=1, scope='convt4', activation_fn=None)
        # This allows the network to also generate one image from scratch,
        # which is useful when regions of the image become unoccluded.
        transformed = [tf.nn.sigmoid(enc7)]

      if stp:
        stp_input0 = tf.reshape(hidden5, [int(batch_size), -1])
        stp_input1 = slim.layers.fully_connected(
            stp_input0, 100, scope='fc_stp')
        transformed += stp_transformation(prev_image, stp_input1, num_masks)
      elif cdna:
        cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
        transformed += cdna_transformation(prev_image, cdna_input, num_masks,
                                           int(color_channels))
      elif dna:
        # Only one mask is supported (more should be unnecessary).
        if num_masks != 1:
          raise ValueError('Only one mask is supported for DNA model.')
        transformed = [dna_transformation(prev_image, enc7)]

      masks = slim.layers.conv2d_transpose(
          enc6, num_masks + 1, 1, stride=1, scope='convt7', activation_fn=None)
      masks = tf.reshape(
          tf.nn.softmax(tf.reshape(masks, [-1, num_masks + 1])),
          [int(batch_size), int(img_height), int(img_width), num_masks + 1])
      mask_list = tf.split(axis=3, num_or_size_splits=num_masks + 1, value=masks)
      output = mask_list[0] * prev_image
      for layer, mask in zip(transformed, mask_list[1:]):
        output += layer * mask
      gen_images.append(output)

      current_state = slim.layers.fully_connected(
          state_action,
          int(current_state.get_shape()[1]),
          scope='state_pred',
          activation_fn=None)
      gen_states.append(current_state)

  return gen_images, gen_states, latent_loss


## Utility functions
def stp_transformation(prev_image, stp_input, num_masks):
  """Apply spatial transformer predictor (STP) to previous image.

  Args:
    prev_image: previous image to be transformed.
    stp_input: hidden layer to be used for computing STN parameters.
    num_masks: number of masks and hence the number of STP transformations.
  Returns:
    List of images transformed by the predicted STP parameters.
  """
  # Only import spatial transformer if needed.
  from spatial_transformer import transformer

  identity_params = tf.convert_to_tensor(
      np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
  transformed = []
  for i in range(num_masks - 1):
    params = slim.layers.fully_connected(
        stp_input, 6, scope='stp_params' + str(i),
        activation_fn=None) + identity_params
    transformed.append(transformer(prev_image, params))

  return transformed


def cdna_transformation(prev_image, cdna_input, num_masks, color_channels):
  """Apply convolutional dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    cdna_input: hidden lyaer to be used for computing CDNA kernels.
    num_masks: the number of masks and hence the number of CDNA transformations.
    color_channels: the number of color channels in the images.
  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  batch_size = int(cdna_input.get_shape()[0])
  height = int(prev_image.get_shape()[1])
  width = int(prev_image.get_shape()[2])

  # Predict kernels using linear function of last hidden layer.
  cdna_kerns = slim.layers.fully_connected(
      cdna_input,
      DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
      scope='cdna_params',
      activation_fn=None)

  # Reshape and normalize.
  cdna_kerns = tf.reshape(
      cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
  cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
  norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keep_dims=True)
  cdna_kerns /= norm_factor

  # Treat the color channel dimension as the batch dimension since the same
  # transformation is applied to each color channel.
  # Treat the batch dimension as the channel dimension so that
  # depthwise_conv2d can apply a different transformation to each sample.
  cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
  cdna_kerns = tf.reshape(cdna_kerns, [DNA_KERN_SIZE, DNA_KERN_SIZE, batch_size, num_masks])
  # Swap the batch and channel dimensions.
  prev_image = tf.transpose(prev_image, [3, 1, 2, 0])

  # Transform image.
  transformed = tf.nn.depthwise_conv2d(prev_image, cdna_kerns, [1, 1, 1, 1], 'SAME')

  # Transpose the dimensions to where they belong.
  transformed = tf.reshape(transformed, [color_channels, height, width, batch_size, num_masks])
  transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
  transformed = tf.unstack(transformed, axis=-1)
  return transformed


def dna_transformation(prev_image, dna_input):
  """Apply dynamic neural advection to previous image.

  Args:
    prev_image: previous image to be transformed.
    dna_input: hidden lyaer to be used for computing DNA transformation.
  Returns:
    List of images transformed by the predicted CDNA kernels.
  """
  # Construct translated images.
  prev_image_pad = tf.pad(prev_image, [[0, 0], [2, 2], [2, 2], [0, 0]])
  image_height = int(prev_image.get_shape()[1])
  image_width = int(prev_image.get_shape()[2])

  inputs = []
  for xkern in range(DNA_KERN_SIZE):
    for ykern in range(DNA_KERN_SIZE):
      inputs.append(
          tf.expand_dims(
              tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                       [-1, image_height, image_width, -1]), [3]))
  inputs = tf.concat(axis=3, values=inputs)

  # Normalize channels to 1.
  kernel = tf.nn.relu(dna_input - RELU_SHIFT) + RELU_SHIFT
  kernel = tf.expand_dims(
      kernel / tf.reduce_sum(
          kernel, [3], keep_dims=True), [4])
  return tf.reduce_sum(kernel * inputs, [3], keep_dims=False)


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
  """Sample batch with specified mix of ground truth and generated data points.

  Args:
    ground_truth_x: tensor of ground-truth data points.
    generated_x: tensor of generated data points.
    batch_size: batch size
    num_ground_truth: number of ground-truth examples to include in batch.
  Returns:
    New batch with num_ground_truth sampled from ground_truth_x and the rest
    from generated_x.
  """
  idx = tf.random_shuffle(tf.range(int(batch_size)))
  ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
  generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

  ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
  generated_examps = tf.gather(generated_x, generated_idx)
  return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                           [ground_truth_examps, generated_examps])
