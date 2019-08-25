# Copyright 2018 The TensorFlow Authors All Rights Reserved.
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

r"""Provides DeepLab model definition and helper functions.

DeepLab is a deep learning system for semantic image segmentation with
the following features:

(1) Atrous convolution to explicitly control the resolution at which
feature responses are computed within Deep Convolutional Neural Networks.

(2) Atrous spatial pyramid pooling (ASPP) to robustly segment objects at
multiple scales with filters at multiple sampling rates and effective
fields-of-views.

(3) ASPP module augmented with image-level feature and batch normalization.

(4) A simple yet effective decoder module to recover the object boundaries.

See the following papers for more details:

"Encoder-Decoder with Atrous Separable Convolution for Semantic Image
Segmentation"
Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam.
(https://arxiv.org/abs1802.02611)

"Rethinking Atrous Convolution for Semantic Image Segmentation,"
Liang-Chieh Chen, George Papandreou, Florian Schroff, Hartwig Adam
(https://arxiv.org/abs/1706.05587)

"DeepLab: Semantic Image Segmentation with Deep Convolutional Nets,
Atrous Convolution, and Fully Connected CRFs",
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L Yuille (* equal contribution)
(https://arxiv.org/abs/1606.00915)

"Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected
CRFs"
Liang-Chieh Chen*, George Papandreou*, Iasonas Kokkinos, Kevin Murphy,
Alan L. Yuille (* equal contribution)
(https://arxiv.org/abs/1412.7062)
"""
import collections
import tensorflow as tf

from deeplab import model
from feelvos import common
from feelvos.utils import embedding_utils
from feelvos.utils import train_utils

slim = tf.contrib.slim


get_branch_logits = model.get_branch_logits
get_extra_layer_scopes = model.get_extra_layer_scopes
multi_scale_logits_v2 = model.multi_scale_logits
refine_by_decoder = model.refine_by_decoder
scale_dimension = model.scale_dimension
split_separable_conv2d = model.split_separable_conv2d

MERGED_LOGITS_SCOPE = model.MERGED_LOGITS_SCOPE
IMAGE_POOLING_SCOPE = model.IMAGE_POOLING_SCOPE
ASPP_SCOPE = model.ASPP_SCOPE
CONCAT_PROJECTION_SCOPE = model.CONCAT_PROJECTION_SCOPE


def predict_labels(images,
                   model_options,
                   image_pyramid=None,
                   reference_labels=None,
                   k_nearest_neighbors=1,
                   embedding_dimension=None,
                   use_softmax_feedback=False,
                   initial_softmax_feedback=None,
                   embedding_seg_feature_dimension=256,
                   embedding_seg_n_layers=4,
                   embedding_seg_kernel_size=7,
                   embedding_seg_atrous_rates=None,
                   also_return_softmax_probabilities=False,
                   num_frames_per_video=None,
                   normalize_nearest_neighbor_distances=False,
                   also_attend_to_previous_frame=False,
                   use_local_previous_frame_attention=False,
                   previous_frame_attention_window_size=9,
                   use_first_frame_matching=True,
                   also_return_embeddings=False,
                   ref_embeddings=None):
  """Predicts segmentation labels.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: An InternalModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.
    reference_labels: A tensor of size [batch, height, width, 1].
      ground truth labels used to perform a nearest neighbor query
    k_nearest_neighbors: Integer, the number of neighbors to use for nearest
      neighbor queries.
    embedding_dimension: Integer, the dimension used for the learned embedding.
    use_softmax_feedback: Boolean, whether to give the softmax predictions of
      the last frame as additional input to the segmentation head.
    initial_softmax_feedback: Float32 tensor, or None. Can be used to
      initialize the softmax predictions used for the feedback loop.
      Typically only useful for inference. Only has an effect if
      use_softmax_feedback is True.
    embedding_seg_feature_dimension: Integer, the dimensionality used in the
      segmentation head layers.
    embedding_seg_n_layers: Integer, the number of layers in the segmentation
      head.
    embedding_seg_kernel_size: Integer, the kernel size used in the
      segmentation head.
    embedding_seg_atrous_rates: List of integers of length
      embedding_seg_n_layers, the atrous rates to use for the segmentation head.
    also_return_softmax_probabilities: Boolean, if true, additionally return
      the softmax probabilities as second return value.
    num_frames_per_video: Integer, the number of frames per video.
    normalize_nearest_neighbor_distances: Boolean, whether to normalize the
      nearest neighbor distances to [0,1] using sigmoid, scale and shift.
    also_attend_to_previous_frame: Boolean, whether to also use nearest
      neighbor attention with respect to the previous frame.
    use_local_previous_frame_attention: Boolean, whether to restrict the
      previous frame attention to a local search window.
      Only has an effect, if also_attend_to_previous_frame is True.
    previous_frame_attention_window_size: Integer, the window size used for
      local previous frame attention, if use_local_previous_frame_attention
      is True.
    use_first_frame_matching: Boolean, whether to extract features by matching
      to the reference frame. This should always be true except for ablation
      experiments.
    also_return_embeddings: Boolean, whether to return the embeddings as well.
    ref_embeddings: Tuple of
      (first_frame_embeddings, previous_frame_embeddings),
      each of shape [batch, height, width, embedding_dimension], or None.

  Returns:
    A dictionary with keys specifying the output_type (e.g., semantic
      prediction) and values storing Tensors representing predictions (argmax
      over channels). Each prediction has size [batch, height, width].
    If also_return_softmax_probabilities is True, the second return value are
      the softmax probabilities.
    If also_return_embeddings is True, it will also return an embeddings
      tensor of shape [batch, height, width, embedding_dimension].

  Raises:
    ValueError: If classification_loss is not softmax, softmax_with_attention,
      nor triplet.
  """
  if (model_options.classification_loss == 'triplet' and
      reference_labels is None):
    raise ValueError('Need reference_labels for triplet loss')

  if model_options.classification_loss == 'softmax_with_attention':
    if embedding_dimension is None:
      raise ValueError('Need embedding_dimension for softmax_with_attention '
                       'loss')
    if reference_labels is None:
      raise ValueError('Need reference_labels for softmax_with_attention loss')
    res = (
        multi_scale_logits_with_nearest_neighbor_matching(
            images,
            model_options=model_options,
            image_pyramid=image_pyramid,
            is_training=False,
            reference_labels=reference_labels,
            clone_batch_size=1,
            num_frames_per_video=num_frames_per_video,
            embedding_dimension=embedding_dimension,
            max_neighbors_per_object=0,
            k_nearest_neighbors=k_nearest_neighbors,
            use_softmax_feedback=use_softmax_feedback,
            initial_softmax_feedback=initial_softmax_feedback,
            embedding_seg_feature_dimension=embedding_seg_feature_dimension,
            embedding_seg_n_layers=embedding_seg_n_layers,
            embedding_seg_kernel_size=embedding_seg_kernel_size,
            embedding_seg_atrous_rates=embedding_seg_atrous_rates,
            normalize_nearest_neighbor_distances=
            normalize_nearest_neighbor_distances,
            also_attend_to_previous_frame=also_attend_to_previous_frame,
            use_local_previous_frame_attention=
            use_local_previous_frame_attention,
            previous_frame_attention_window_size=
            previous_frame_attention_window_size,
            use_first_frame_matching=use_first_frame_matching,
            also_return_embeddings=also_return_embeddings,
            ref_embeddings=ref_embeddings
        ))
    if also_return_embeddings:
      outputs_to_scales_to_logits, embeddings = res
    else:
      outputs_to_scales_to_logits = res
      embeddings = None
  else:
    outputs_to_scales_to_logits = multi_scale_logits_v2(
        images,
        model_options=model_options,
        image_pyramid=image_pyramid,
        is_training=False,
        fine_tune_batch_norm=False)

  predictions = {}
  for output in sorted(outputs_to_scales_to_logits):
    scales_to_logits = outputs_to_scales_to_logits[output]
    original_logits = scales_to_logits[MERGED_LOGITS_SCOPE]
    if isinstance(original_logits, list):
      assert len(original_logits) == 1
      original_logits = original_logits[0]
    logits = tf.image.resize_bilinear(original_logits, tf.shape(images)[1:3],
                                      align_corners=True)
    if model_options.classification_loss in ('softmax',
                                             'softmax_with_attention'):
      predictions[output] = tf.argmax(logits, 3)
    elif model_options.classification_loss == 'triplet':
      # to keep this fast, we do the nearest neighbor assignment on the
      # resolution at which the embedding is extracted and scale the result up
      # afterwards
      embeddings = original_logits
      reference_labels_logits_size = tf.squeeze(
          tf.image.resize_nearest_neighbor(
              reference_labels[tf.newaxis],
              train_utils.resolve_shape(embeddings)[1:3],
              align_corners=True), axis=0)
      nn_labels = embedding_utils.assign_labels_by_nearest_neighbors(
          embeddings[0], embeddings[1:], reference_labels_logits_size,
          k_nearest_neighbors)
      predictions[common.OUTPUT_TYPE] = tf.image.resize_nearest_neighbor(
          nn_labels, tf.shape(images)[1:3], align_corners=True)
    else:
      raise ValueError(
          'Only support softmax, triplet, or softmax_with_attention for '
          'classification_loss.')

  if also_return_embeddings:
    assert also_return_softmax_probabilities
    return predictions, tf.nn.softmax(original_logits, axis=-1), embeddings
  elif also_return_softmax_probabilities:
    return predictions, tf.nn.softmax(original_logits, axis=-1)
  else:
    return predictions


def multi_scale_logits_with_nearest_neighbor_matching(
    images,
    model_options,
    image_pyramid,
    clone_batch_size,
    reference_labels,
    num_frames_per_video,
    embedding_dimension,
    max_neighbors_per_object,
    weight_decay=0.0001,
    is_training=False,
    fine_tune_batch_norm=False,
    k_nearest_neighbors=1,
    use_softmax_feedback=False,
    initial_softmax_feedback=None,
    embedding_seg_feature_dimension=256,
    embedding_seg_n_layers=4,
    embedding_seg_kernel_size=7,
    embedding_seg_atrous_rates=None,
    normalize_nearest_neighbor_distances=False,
    also_attend_to_previous_frame=False,
    damage_initial_previous_frame_mask=False,
    use_local_previous_frame_attention=False,
    previous_frame_attention_window_size=9,
    use_first_frame_matching=True,
    also_return_embeddings=False,
    ref_embeddings=None):
  """Gets the logits for multi-scale inputs using nearest neighbor attention.

  Adjusted version of multi_scale_logits_v2 to support nearest neighbor
  attention and a variable number of classes for each element of the batch.
  The returned logits are all downsampled (due to max-pooling layers)
  for both training and evaluation.

  Args:
    images: A tensor of size [batch, height, width, channels].
    model_options: A ModelOptions instance to configure models.
    image_pyramid: Input image scales for multi-scale feature extraction.
    clone_batch_size: Integer, the number of videos on a batch.
    reference_labels: The segmentation labels of the reference frame on which
      attention is applied.
    num_frames_per_video: Integer, the number of frames per video.
    embedding_dimension: Integer, the dimension of the embedding.
    max_neighbors_per_object: Integer, the maximum number of candidates
      for the nearest neighbor query per object after subsampling.
      Can be 0 for no subsampling.
    weight_decay: The weight decay for model variables.
    is_training: Is training or not.
    fine_tune_batch_norm: Fine-tune the batch norm parameters or not.
    k_nearest_neighbors: Integer, the number of nearest neighbors to use.
    use_softmax_feedback: Boolean, whether to give the softmax predictions of
      the last frame as additional input to the segmentation head.
    initial_softmax_feedback: List of Float32 tensors, or None.
      Can be used to initialize the softmax predictions used for the feedback
      loop. Only has an effect if use_softmax_feedback is True.
    embedding_seg_feature_dimension: Integer, the dimensionality used in the
      segmentation head layers.
    embedding_seg_n_layers: Integer, the number of layers in the segmentation
      head.
    embedding_seg_kernel_size: Integer, the kernel size used in the
      segmentation head.
    embedding_seg_atrous_rates: List of integers of length
      embedding_seg_n_layers, the atrous rates to use for the segmentation head.
    normalize_nearest_neighbor_distances: Boolean, whether to normalize the
      nearest neighbor distances to [0,1] using sigmoid, scale and shift.
    also_attend_to_previous_frame: Boolean, whether to also use nearest
      neighbor attention with respect to the previous frame.
    damage_initial_previous_frame_mask: Boolean, whether to artificially damage
      the initial previous frame mask. Only has an effect if
      also_attend_to_previous_frame is True.
    use_local_previous_frame_attention: Boolean, whether to restrict the
      previous frame attention to a local search window.
      Only has an effect, if also_attend_to_previous_frame is True.
    previous_frame_attention_window_size: Integer, the window size used for
      local previous frame attention, if use_local_previous_frame_attention
      is True.
    use_first_frame_matching: Boolean, whether to extract features by matching
      to the reference frame. This should always be true except for ablation
      experiments.
    also_return_embeddings: Boolean, whether to return the embeddings as well.
    ref_embeddings: Tuple of
      (first_frame_embeddings, previous_frame_embeddings),
      each of shape [batch, height, width, embedding_dimension], or None.

  Returns:
    outputs_to_scales_to_logits: A map of maps from output_type (e.g.,
      semantic prediction) to a dictionary of multi-scale logits names to
      logits. For each output_type, the dictionary has keys which
      correspond to the scales and values which correspond to the logits.
      For example, if `scales` equals [1.0, 1.5], then the keys would
      include 'merged_logits', 'logits_1.00' and 'logits_1.50'.
    If also_return_embeddings is True, it will also return an embeddings
      tensor of shape [batch, height, width, embedding_dimension].

  Raises:
    ValueError: If model_options doesn't specify crop_size and its
      add_image_level_feature = True, since add_image_level_feature requires
      crop_size information.
  """
  # Setup default values.
  if not image_pyramid:
    image_pyramid = [1.0]
  crop_height = (
      model_options.crop_size[0]
      if model_options.crop_size else tf.shape(images)[1])
  crop_width = (
      model_options.crop_size[1]
      if model_options.crop_size else tf.shape(images)[2])

  # Compute the height, width for the output logits.
  if model_options.decoder_output_stride:
    logits_output_stride = min(model_options.decoder_output_stride)
  else:
    logits_output_stride = model_options.output_stride
  logits_height = scale_dimension(
      crop_height,
      max(1.0, max(image_pyramid)) / logits_output_stride)
  logits_width = scale_dimension(
      crop_width,
      max(1.0, max(image_pyramid)) / logits_output_stride)

  # Compute the logits for each scale in the image pyramid.
  outputs_to_scales_to_logits = {
      k: {}
      for k in model_options.outputs_to_num_classes
  }

  for image_scale in image_pyramid:
    if image_scale != 1.0:
      scaled_height = scale_dimension(crop_height, image_scale)
      scaled_width = scale_dimension(crop_width, image_scale)
      scaled_crop_size = [scaled_height, scaled_width]
      scaled_images = tf.image.resize_bilinear(
          images, scaled_crop_size, align_corners=True)
      scaled_reference_labels = tf.image.resize_nearest_neighbor(
          reference_labels, scaled_crop_size, align_corners=True
      )
      if model_options.crop_size is None:
        scaled_crop_size = None
      if model_options.crop_size:
        scaled_images.set_shape([None, scaled_height, scaled_width, 3])
    else:
      scaled_crop_size = model_options.crop_size
      scaled_images = images
      scaled_reference_labels = reference_labels

    updated_options = model_options._replace(crop_size=scaled_crop_size)
    res = embedding_utils.get_logits_with_matching(
        scaled_images,
        updated_options,
        weight_decay=weight_decay,
        reuse=tf.AUTO_REUSE,
        is_training=is_training,
        fine_tune_batch_norm=fine_tune_batch_norm,
        reference_labels=scaled_reference_labels,
        batch_size=clone_batch_size,
        num_frames_per_video=num_frames_per_video,
        embedding_dimension=embedding_dimension,
        max_neighbors_per_object=max_neighbors_per_object,
        k_nearest_neighbors=k_nearest_neighbors,
        use_softmax_feedback=use_softmax_feedback,
        initial_softmax_feedback=initial_softmax_feedback,
        embedding_seg_feature_dimension=embedding_seg_feature_dimension,
        embedding_seg_n_layers=embedding_seg_n_layers,
        embedding_seg_kernel_size=embedding_seg_kernel_size,
        embedding_seg_atrous_rates=embedding_seg_atrous_rates,
        normalize_nearest_neighbor_distances=
        normalize_nearest_neighbor_distances,
        also_attend_to_previous_frame=also_attend_to_previous_frame,
        damage_initial_previous_frame_mask=damage_initial_previous_frame_mask,
        use_local_previous_frame_attention=use_local_previous_frame_attention,
        previous_frame_attention_window_size=
        previous_frame_attention_window_size,
        use_first_frame_matching=use_first_frame_matching,
        also_return_embeddings=also_return_embeddings,
        ref_embeddings=ref_embeddings
    )
    if also_return_embeddings:
      outputs_to_logits, embeddings = res
    else:
      outputs_to_logits = res
      embeddings = None

    # Resize the logits to have the same dimension before merging.
    for output in sorted(outputs_to_logits):
      if isinstance(outputs_to_logits[output], collections.Sequence):
        outputs_to_logits[output] = [tf.image.resize_bilinear(
            x, [logits_height, logits_width], align_corners=True)
                                     for x in outputs_to_logits[output]]
      else:
        outputs_to_logits[output] = tf.image.resize_bilinear(
            outputs_to_logits[output], [logits_height, logits_width],
            align_corners=True)

    # Return when only one input scale.
    if len(image_pyramid) == 1:
      for output in sorted(model_options.outputs_to_num_classes):
        outputs_to_scales_to_logits[output][
            MERGED_LOGITS_SCOPE] = outputs_to_logits[output]
      if also_return_embeddings:
        return outputs_to_scales_to_logits, embeddings
      else:
        return outputs_to_scales_to_logits

    # Save logits to the output map.
    for output in sorted(model_options.outputs_to_num_classes):
      outputs_to_scales_to_logits[output][
          'logits_%.2f' % image_scale] = outputs_to_logits[output]

  # Merge the logits from all the multi-scale inputs.
  for output in sorted(model_options.outputs_to_num_classes):
    # Concatenate the multi-scale logits for each output type.
    all_logits = [
        [tf.expand_dims(l, axis=4)]
        for logits in outputs_to_scales_to_logits[output].values()
        for l in logits
    ]
    transposed = map(list, zip(*all_logits))
    all_logits = [tf.concat(t, 4) for t in transposed]
    merge_fn = (
        tf.reduce_max
        if model_options.merge_method == 'max' else tf.reduce_mean)
    outputs_to_scales_to_logits[output][MERGED_LOGITS_SCOPE] = [merge_fn(
        l, axis=4) for l in all_logits]

  if also_return_embeddings:
    return outputs_to_scales_to_logits, embeddings
  else:
    return outputs_to_scales_to_logits
