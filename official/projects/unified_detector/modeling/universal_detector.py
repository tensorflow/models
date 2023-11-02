# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

"""Universal detector implementation."""

from typing import Any, Dict, Optional, Sequence, Tuple, Union

import gin
import tensorflow as tf, tf_keras

from deeplab2 import config_pb2
from deeplab2.model.decoder import max_deeplab as max_deeplab_head
from deeplab2.model.encoder import axial_resnet_instances
from deeplab2.model.loss import matchers_ops
from official.legacy.transformer import transformer
from official.projects.unified_detector.utils import typing
from official.projects.unified_detector.utils import utilities


EPSILON = 1e-6


@gin.configurable
def universal_detection_loss_weights(
    loss_segmentation_word: float = 1e0,
    loss_inst_dist: float = 1e0,
    loss_mask_id: float = 1e-4,
    loss_pq: float = 3e0,
    loss_para: float = 1e0) -> Dict[str, float]:
  """A function that returns a dict for the weights of loss terms."""
  return {
      "loss_segmentation_word": loss_segmentation_word,
      "loss_inst_dist": loss_inst_dist,
      "loss_mask_id": loss_mask_id,
      "loss_pq": loss_pq,
      "loss_para": loss_para,
  }


@gin.configurable
class LayerNorm(tf_keras.layers.LayerNormalization):
  """A wrapper to allow passing the `training` argument.

  The normalization layers in the MaX-DeepLab implementation are passed with
  the `training` argument. This wrapper enables the usage of LayerNorm.
  """

  def call(self,
           inputs: tf.Tensor,
           training: Optional[bool] = None) -> tf.Tensor:
    del training
    return super().call(inputs)


@gin.configurable
def get_max_deep_lab_backbone(num_slots: int = 128):
  return axial_resnet_instances.get_model(
      "max_deeplab_s",
      bn_layer=LayerNorm,
      block_group_config={
          "drop_path_schedule": "linear",
          "axial_use_recompute_grad": False
      },
      backbone_use_transformer_beyond_stride=16,
      extra_decoder_use_transformer_beyond_stride=16,
      num_mask_slots=num_slots,
      max_num_mask_slots=num_slots)


@gin.configurable
class UniversalDetector(tf_keras.layers.Layer):
  """Univeral Detector."""
  loss_items = ("loss_pq", "loss_inst_dist", "loss_para", "loss_mask_id",
                "loss_segmentation_word")

  def __init__(self,
               backbone_fn: tf_keras.layers.Layer = get_max_deep_lab_backbone,
               mask_threshold: float = 0.4,
               class_threshold: float = 0.5,
               filter_area: float = 32,
               **kwargs: Any):
    """Constructor.

    Args:
      backbone_fn: The function to initialize a backbone.
      mask_threshold: Masks are thresholded with this value.
      class_threshold: Classification heads are thresholded with this value.
      filter_area: In inference, detections with area smaller than this
        threshold will be removed.
      **kwargs: other keyword arguments passed to the base class.
    """
    super().__init__(**kwargs)

    # Model
    self._backbone_fn = backbone_fn()
    self._decoder = _get_decoder_head()
    self._class_embed_head, self._para_embed_head = _get_embed_head()
    self._para_head, self._para_proj = _get_para_head()

    # Losses
    # self._max_deeplab_loss = _get_max_deeplab_loss()
    self._loss_weights = universal_detection_loss_weights()

    # Post-processing
    self._mask_threshold = mask_threshold
    self._class_threshold = class_threshold
    self._filter_area = filter_area

  def _preprocess_labels(self, labels: typing.TensorDict):
    # Preprocessing
    # Converted the integer mask to one-hot embedded masks.
    num_instances = utilities.resolve_shape(
        labels["instance_labels"]["masks_sizes"])[1]
    labels["instance_labels"]["masks"] = tf.one_hot(
        labels["instance_labels"]["masks"],
        depth=num_instances,
        axis=1,
        dtype=tf.float32)  # (B, N, H, W)

  def compute_losses(
      self, labels: typing.NestedTensorDict, outputs: typing.NestedTensorDict
  ) -> Tuple[tf.Tensor, typing.NestedTensorDict]:
    """Computes the loss.

    Args:
      labels: A dictionary of ground-truth labels.
      outputs: Output from self.call().

    Returns:
      A scalar total loss tensor and a dictionary for individual losses.
    """
    loss_dict = {}

    self._preprocess_labels(labels)

    # Main loss: PQ loss.
    _entity_mask_loss(loss_dict, labels["instance_labels"],
                      outputs["instance_output"])
    # Auxiliary loss 1: semantic loss
    _semantic_loss(loss_dict, labels["segmentation_output"],
                   outputs["segmentation_output"])
    # Auxiliary loss 2: instance discrimination
    _instance_discrimination_loss(loss_dict, labels["instance_labels"], outputs)
    # Auxiliary loss 3: mask id
    _mask_id_xent_loss(loss_dict, labels["instance_labels"], outputs)
    # Auxiliary loss 4: paragraph grouping
    _paragraph_grouping_loss(loss_dict, labels, outputs)

    weighted_loss = [self._loss_weights[k] * v for k, v in loss_dict.items()]
    total_loss = sum(weighted_loss)
    return total_loss, loss_dict

  def call(self,
           features: typing.TensorDict,
           training: bool = False) -> typing.NestedTensorDict:
    """Forward pass of the model.

    Args:
      features: The input features: {"images": tf.Tensor}. Shape = [B, H, W, C]
      training: Whether it's training mode.

    Returns:
      A dictionary of output with this structure:
        {
          "max_deep_lab": {
            All the max deeplab outputs are here, including both backbone and
            decoder.
          }
          "segmentation_output": {
            "word_score": tf.Tensor, [B, h, w],
          }
          "instance_output": {
            "cls_logits": tf.Tensor, [B, N, C],
            "mask_id_logits": tf.Tensor, [B, H, W, N],
            "cls_prob":  tf.Tensor, [B, N, C],
            "mask_id_prob": tf.Tensor, [B, H, W, N],
          }
          "postprocessed": {
            "classes": A (B, N) tensor for the class ids. Zero for non-firing
              slots.
            "binary_masks": A (B, H, W, N) tensor for the N binary masks. Masks
              for void cls are set to zero.
            "confidence": A (B, N) float tensor for the confidence of "classes".
            "mask_area": A (B, N) float tensor for the area of each mask.
          }
          "transformer_group_feature": (B, N, C) float tensor (normalized),
          "para_affinity": (B, N, N) float tensor.
        }

      Class-0 is for void. Class-(C-1) is for background. Class-1~(C-2) is for
      valid classes.
    """
    # backbone
    backbone_output = self._backbone_fn(features["images"], training)
    # split instance embedding and paragraph embedding;
    # then perform paragraph grouping
    para_fts = self._get_para_outputs(backbone_output, training)
    affinity = tf.linalg.matmul(para_fts, para_fts, transpose_b=True)
    # text detection head
    decoder_output = self._decoder(backbone_output, training)
    output_dict = {
        "max_deep_lab": decoder_output,
        "transformer_group_feature": para_fts,
        "para_affinity": affinity,
    }
    input_shape = utilities.resolve_shape(features["images"])
    self._get_semantic_outputs(output_dict, input_shape)
    self._get_instance_outputs(output_dict, input_shape)
    self._postprocess(output_dict)

    return output_dict

  def _get_para_outputs(self, outputs: typing.TensorDict,
                        training: bool) -> tf.Tensor:
    """Apply the paragraph head.

    This function first splits the features for instance classification and
    instance grouping. Then, the additional grouping branch (transformer layers)
    is applied to further encode the grouping features. Finally, a tensor of
    normalized grouping features is returned.

    Args:
      outputs: output dictionary from the backbone.
      training: training / eval mode mark.

    Returns:
      The normalized paragraph embedding vector of shape (B, N, C).
    """
    # Project the object embeddings into classification feature and grouping
    # feature.
    fts = outputs["transformer_class_feature"]  # B,N,C
    class_feature = self._class_embed_head(fts, training)
    group_feature = self._para_embed_head(fts, training)
    outputs["transformer_class_feature"] = class_feature
    outputs["transformer_group_feature"] = group_feature

    # Feed the grouping features into additional group encoding branch.
    # First we need to build the attention_bias which is used the standard
    # transformer encoder.
    input_shape = utilities.resolve_shape(group_feature)
    b = input_shape[0]
    n = int(input_shape[1])
    seq_len = tf.constant(n, shape=(b,))
    padding_mask = utilities.get_padding_mask_from_valid_lengths(
        seq_len, n, tf.float32)
    attention_bias = utilities.get_transformer_attention_bias(padding_mask)
    group_feature = self._para_proj(
        self._para_head(group_feature, attention_bias, None, training))
    return tf.math.l2_normalize(group_feature, axis=-1)

  def _get_semantic_outputs(self, outputs: typing.NestedTensorDict,
                            input_shape: tf.TensorShape):
    """Add `segmentation_output` to outputs.

    Args:
      outputs: A dictionary of outputs.
      input_shape: The shape of the input images.
    """
    h, w = input_shape[1:3]
    # B, H/4, W/4, C
    semantic_logits = outputs["max_deep_lab"]["semantic_logits"]
    textness, unused_logits = tf.split(semantic_logits, [2, -1], -1)
    # Channel[0:2], textness. c0: non-textness, c1: textness.
    word_score = tf.nn.softmax(textness, -1, "word_score")[:, :, :, 1:2]
    word_score = tf.squeeze(tf.image.resize(word_score, (h, w)), -1)
    # Channel[2:] not used yet
    outputs["segmentation_output"] = {"word_score": word_score}

  def _get_instance_outputs(self, outputs: typing.NestedTensorDict,
                            input_shape: tf.TensorShape):
    """Add `instance_output` to outputs.

    Args:
      outputs: A dictionary of outputs.
      input_shape: The shape of the input images.
    These following fields are added to outputs["instance_output"]:
      "cls_logits": tf.Tensor, [B, N, C].
      "mask_id_logits": tf.Tensor, [B, H, W, N].
      "cls_prob":  tf.Tensor, [B, N, C], softmax probability.
      "mask_id_prob": tf.Tensor, [B, H, W, N], softmax probability. They are
        used in training. Masks are all resized to full resolution.
    """
    # Get instance_output
    h, w = input_shape[1:3]
    ## Classes
    class_logits = outputs["max_deep_lab"]["transformer_class_logits"]
    # The MaX-DeepLab repo uses the last logit for void; but we use 0.
    # Therefore we shift the logits here.
    class_logits = tf.roll(class_logits, shift=1, axis=-1)
    class_prob = tf.nn.softmax(class_logits)

    ## Masks
    mask_id_logits = outputs["max_deep_lab"]["pixel_space_mask_logits"]
    mask_id_prob = tf.nn.softmax(mask_id_logits)
    mask_id_logits = tf.image.resize(mask_id_logits, (h, w))
    mask_id_prob = tf.image.resize(mask_id_prob, (h, w))
    outputs["instance_output"] = {
        "cls_logits": class_logits,
        "mask_id_logits": mask_id_logits,
        "cls_prob": class_prob,
        "mask_id_prob": mask_id_prob,
    }

  def _postprocess(self, outputs: typing.NestedTensorDict):
    """Post-process (filtering) the outputs.

    Args:
      outputs: A dictionary of outputs.
    These following fields are added to outputs["postprocessed"]:
      "classes": A (B,N) integer tensor for the class ids.
      "binary_masks": A (B, H, W, N) tensor for the N binarized 0/1 masks. Masks
        for void cls are set to zero.
      "confidence": A (B, N) float tensor for the confidence of "classes".
      "mask_area": A (B, N) float tensor for the area of each mask. They are
        used in inference / visualization.
    """
    # Get postprocessed outputs
    outputs["postprocessed"] = {}

    ## Masks:
    mask_id_prob = outputs["instance_output"]["mask_id_prob"]
    mask_max_prob = tf.reduce_max(mask_id_prob, axis=-1, keepdims=True)
    thresholded_binary_masks = tf.cast(
        tf.math.logical_and(
            tf.equal(mask_max_prob, mask_id_prob),
            tf.greater_equal(mask_max_prob, self._mask_threshold)), tf.float32)
    area = tf.reduce_sum(thresholded_binary_masks, axis=(1, 2))  # (B, N)
    ## Classification:
    cls_prob = outputs["instance_output"]["cls_prob"]
    cls_max_prob = tf.reduce_max(cls_prob, axis=-1)  # B, N
    cls_max_id = tf.cast(tf.argmax(cls_prob, axis=-1), tf.float32)  # B, N

    ## filtering
    c = utilities.resolve_shape(cls_prob)[2]
    non_void = tf.reduce_all(
        tf.stack(
            [
                tf.greater_equal(area, self._filter_area),  # mask large enough.
                tf.not_equal(cls_max_id, 0),  # class-0 is for non-object.
                tf.not_equal(cls_max_id,
                             c - 1),  # class-(c-1) is for background (last).
                tf.greater_equal(cls_max_prob,
                                 self._class_threshold)  # prob >= thr
            ],
            axis=-1),
        axis=-1)
    non_void = tf.cast(non_void, tf.float32)

    # Storing
    outputs["postprocessed"]["classes"] = tf.cast(cls_max_id * non_void,
                                                  tf.int32)
    b, n = utilities.resolve_shape(non_void)
    outputs["postprocessed"]["binary_masks"] = (
        thresholded_binary_masks * tf.reshape(non_void, (b, 1, 1, n)))
    outputs["postprocessed"]["confidence"] = cls_max_prob
    outputs["postprocessed"]["mask_area"] = area

  def _coloring(self, masks: tf.Tensor) -> tf.Tensor:
    """Coloring segmentation masks.

    Used in visualization.

    Args:
      masks: A float binary tensor of shape (B, H, W, N), representing `B`
        samples, with `N` masks of size `H*W` each. Each of the `N` masks will
        be assigned a random color.

    Returns:
      A (b, h, w, 3) float tensor in [0., 1.] for the coloring result.
    """
    b, h, w, n = utilities.resolve_shape(masks)
    palette = tf.random.uniform((1, n, 3), 0.5, 1.)
    colored = tf.reshape(
        tf.matmul(tf.reshape(masks, (b, -1, n)), palette), (b, h, w, 3))
    return colored

  def visualize(self,
                outputs: typing.NestedTensorDict,
                labels: Optional[typing.TensorDict] = None):
    """Visualizes the outputs and labels.

    Args:
      outputs: A dictionary of outputs.
      labels: A dictionary of labels.
    The following dict is added to outputs["visualization"]: {
        "instance": {
          "pred": A (B, H, W, 3) tensor for the visualized map in [0,1].
          "gt": A (B, H, W, 3) tensor for the visualized map in [0,1], if labels
            is present.
          "concat": Concatenation of "prediction" and "gt" along width axis, if
            labels is present. }
        "seg-text": {... Similar to above, but the shape is (B, H, W, 1).} } All
          of these tensors have a rank of 4 (B, H, W, C).
    """

    outputs["visualization"] = {}
    # 1. prediction
    # 1.1 instance mask
    binary_masks = outputs["postprocessed"]["binary_masks"]
    outputs["visualization"]["instance"] = {
        "pred": self._coloring(binary_masks),
    }
    # 1.2 text-seg
    outputs["visualization"]["seg-text"] = {
        "pred":
            tf.expand_dims(outputs["segmentation_output"]["word_score"], -1),
    }

    # 2. labels
    if labels is not None:
      # 2.1 instance mask
      # (B, N, H, W) -> (B, H, W, N); the first one is bkg so removed.
      gt_masks = tf.transpose(labels["instance_labels"]["masks"][:, 1:],
                              (0, 2, 3, 1))
      outputs["visualization"]["instance"]["gt"] = self._coloring(gt_masks)
      # 2.2 text-seg
      outputs["visualization"]["seg-text"]["gt"] = tf.expand_dims(
          labels["segmentation_output"]["gt_word_score"], -1)

      # 3. concat
      for v in outputs["visualization"].values():
        # Resize to make the size align. The prediction always has stride=1
        # resolution, so we make gt align with pred instead of vice versa.
        v["concat"] = tf.concat(
            [v["pred"],
             tf.image.resize(v["gt"],
                             tf.shape(v["pred"])[1:3])],
            axis=2)

  @tf.function
  def serve(self, image_tensor: tf.Tensor) -> typing.NestedTensorDict:
    """Method to be exported for SavedModel.

    Args:
      image_tensor: A float32 normalized tensor representing an image of shape
        [1, height, width, channels].

    Returns:
      Dict of output:
        classes: (B, N) int32 tensor == o["postprocessed"]["classes"]
        masks: (B, H, W, N) float32 tensor == o["postprocessed"]["binary_masks"]
        groups: (B, N, N) float32 tensor == o["para_affinity"]
        confidence: A (B, N) float tensor == o["postprocessed"]["confidence"]
        mask_area: A (B, N) float tensor == o["postprocessed"]["mask_area"]
    """
    features = {"images": image_tensor}
    nn_outputs = self(features, False)
    outputs = {
        "classes": nn_outputs["postprocessed"]["classes"],
        "masks": nn_outputs["postprocessed"]["binary_masks"],
        "confidence": nn_outputs["postprocessed"]["confidence"],
        "mask_area": nn_outputs["postprocessed"]["mask_area"],
        "groups": nn_outputs["para_affinity"],
    }
    return outputs


@gin.configurable()
def _get_decoder_head(
    atrous_rates: Sequence[int] = (6, 12, 18),
    pixel_space_dim: int = 128,
    pixel_space_intermediate: int = 256,
    low_level: Sequence[Dict[str, Union[str, int]]] = ({
        "feature_key": "res3",
        "channels_project": 64,
    }, {
        "feature_key": "res2",
        "channels_project": 32,
    }),
    num_classes=3,
    aux_sem_intermediate=256,
    norm_fn=tf_keras.layers.BatchNormalization,
) -> max_deeplab_head.MaXDeepLab:
  """Get the MaX-DeepLab prediction head.

  Args:
    atrous_rates: Dilation rate for astrou conv in the semantic head.
    pixel_space_dim: The dimension for the final panoptic features.
    pixel_space_intermediate: The dimension for the layer before
      `pixel_space_dim` (i.e. the separable 5x5 layer).
    low_level: A list of dicts for the feature pyramid in forming the semantic
      output. Each dict represents one skip-path from the backbone.
    num_classes: Number of classes (entities + bkg) including void. For example,
      if we only want to detect word, then `num_classes` = 3 (1 for word, 1 for
      bkg, and 1 for void).
    aux_sem_intermediate: Similar to `pixel_space_intermediate`, but for the
      auxiliary semantic output head.
    norm_fn: The normalization function used in the head.

  Returns:
    A MaX-DeepLab decoder head (as a keras layer).
  """

  # Initialize the configs.
  configs = config_pb2.ModelOptions()
  configs.decoder.feature_key = "feature_semantic"
  configs.decoder.atrous_rates.extend(atrous_rates)
  configs.max_deeplab.pixel_space_head.output_channels = pixel_space_dim
  configs.max_deeplab.pixel_space_head.head_channels = pixel_space_intermediate
  for low_level_config in low_level:
    low_level_ = configs.max_deeplab.auxiliary_low_level.add()
    low_level_.feature_key = low_level_config["feature_key"]
    low_level_.channels_project = low_level_config["channels_project"]
  configs.max_deeplab.auxiliary_semantic_head.output_channels = num_classes
  configs.max_deeplab.auxiliary_semantic_head.head_channels = aux_sem_intermediate

  return max_deeplab_head.MaXDeepLab(configs.decoder,
                                     configs.max_deeplab, 0, norm_fn)


class PseudoLayer(tf_keras.layers.Layer):
  """Pseudo layer for ablation study.

  The `call()` function has the same argument signature as a transformer
  encoder stack. `unused_ph1` and `unused_ph2` are place holders for this
  purpose. When studying the effectiveness of using transformer as the
  grouping branch, we can use this PseudoLayer to replace the transformer to
  use as a no-transformer baseline.

  To use a single projection layer instead of transformer, simply set `extra_fc`
  to True.
  """

  def __init__(self, extra_fc: bool):
    super().__init__(name="extra_fc")
    self._extra_fc = extra_fc
    if extra_fc:
      self._layer = tf_keras.Sequential([
          tf_keras.layers.Dense(256, activation="relu"),
          tf_keras.layers.LayerNormalization(),
      ])

  def call(self,
           fts: tf.Tensor,
           unused_ph1: Optional[tf.Tensor],
           unused_ph2: Optional[tf.Tensor],
           training: Optional[bool] = None) -> tf.Tensor:
    """See base class."""
    if self._extra_fc:
      return self._layer(fts, training)
    return fts


@gin.configurable()
def _get_embed_head(
    dimension=256,
    norm_fn=tf_keras.layers.BatchNormalization
) -> Tuple[tf_keras.Sequential, tf_keras.Sequential]:
  """Projection layers to get instance & grouping features."""
  instance_head = tf_keras.Sequential([
      tf_keras.layers.Dense(dimension, use_bias=False),
      norm_fn(),
      tf_keras.layers.ReLU(),
  ])
  grouping_head = tf_keras.Sequential([
      tf_keras.layers.Dense(dimension, use_bias=False),
      norm_fn(),
      tf_keras.layers.ReLU(),
  ])
  return instance_head, grouping_head


@gin.configurable()
def _get_para_head(
    dimension=128,
    num_layer=3,
    extra_fc=False) -> Tuple[tf_keras.layers.Layer, tf_keras.layers.Layer]:
  """Get the additional para head.

  Args:
    dimension: the dimension of the final output.
    num_layer: the number of transformer layer.
    extra_fc: Whether an extra single fully-connected layer is used, when
      num_layer=0.

  Returns:
    an encoder and a projection layer for the grouping features.
  """
  if num_layer > 0:
    encoder = transformer.EncoderStack(
        params={
            "hidden_size": 256,
            "num_hidden_layers": num_layer,
            "num_heads": 4,
            "filter_size": 512,
            "initializer_gain": 1.0,
            "attention_dropout": 0.1,
            "relu_dropout": 0.1,
            "layer_postprocess_dropout": 0.1,
            "allow_ffn_pad": True,
        })
  else:
    encoder = PseudoLayer(extra_fc)
  dense = tf_keras.layers.Dense(dimension)
  return encoder, dense


def _dice_sim(pred: tf.Tensor, ground_truth: tf.Tensor) -> tf.Tensor:
  """Dice Coefficient for mask similarity.

  Args:
    pred: The predicted mask. [B, N, H, W], in [0, 1].
    ground_truth: The ground-truth mask. [B, N, H, W], in [0, 1] or {0, 1}.

  Returns:
    A matrix for the losses: m[b, i, j] is the dice similarity between pred `i`
    and gt `j` in batch `b`.
  """
  b, n = utilities.resolve_shape(pred)[:2]
  ground_truth = tf.reshape(
      tf.transpose(ground_truth, (0, 2, 3, 1)), (b, -1, n))  # B, HW, N
  pred = tf.reshape(pred, (b, n, -1))  # B, N, HW
  numerator = tf.matmul(pred, ground_truth) * 2.
  # TODO(longshangbang): The official implementation does not square the scores.
  # Need to do experiment to determine which one is better.
  denominator = (
      tf.math.reduce_sum(tf.math.square(ground_truth), 1, keepdims=True) +
      tf.math.reduce_sum(tf.math.square(pred), 2, keepdims=True))
  return (numerator + EPSILON) / (denominator + EPSILON)


def _semantic_loss(
    loss_dict: Dict[str, tf.Tensor],
    labels: tf.Tensor,
    outputs: tf.Tensor,
):
  """Auxiliary semantic loss.

  Currently, these losses are added:
    (1) text/non-text heatmap

  Args:
    loss_dict: A dictionary for the loss. The values are loss scalars.
    labels: The label dictionary containing:
      `gt_word_score`: (B, H, W) tensor for the text/non-text map.
    outputs: The output dictionary containing:
      `word_score`: (B, H, W) prediction tensor for `gt_word_score`
  """
  pred = tf.expand_dims(outputs["word_score"], 1)
  gt = tf.expand_dims(labels["gt_word_score"], 1)
  loss_dict["loss_segmentation_word"] = 1. - tf.reduce_mean(_dice_sim(pred, gt))


@gin.configurable
def _entity_mask_loss(loss_dict: Dict[str, tf.Tensor],
                      labels: tf.Tensor,
                      outputs: tf.Tensor,
                      alpha: float = gin.REQUIRED):
  """PQ loss for entity-mask training.

  This method adds the PQ loss term to loss_dict directly. The match result will
  also be stored in outputs (As a [B, N_pred, N_gt] float tensor).

  Args:
    loss_dict: A dictionary for the loss. The values are loss scalars.
    labels: A dict containing: `num_instance` - (B,) `masks` - (B, N, H, W)
      `classes` - (B, N)
    outputs: A dict containing:
      `cls_prob`: (B, N, C)
      `mask_id_prob`: (B, H, W, N)
      `cls_logits`: (B, N, C)
      `mask_id_logits`: (B, H, W, N)
    alpha: Weight for pos/neg balance.
  """
  # Classification score: (B, N, N)
  # in batch b, the probability of prediction i being class of gt j, i.e.:
  # score[b, i, j] = pred_cls[b, i, gt_cls[b, j]]
  gt_cls = labels["classes"]  # (B, N)
  pred_cls = outputs["cls_prob"]  # (B, N, C)
  b, n = utilities.resolve_shape(pred_cls)[:2]
  # indices[b, i, j] = gt_cls[b, j]
  indices = tf.tile(tf.expand_dims(gt_cls, 1), (1, n, 1))
  cls_score = tf.gather(pred_cls, tf.cast(indices, tf.int32), batch_dims=2)

  # Mask score (dice): (B, N, N)
  # mask_score[b, i, j]: dice-similarity for pred i and gt j in batch b.
  mask_score = _dice_sim(
      tf.transpose(outputs["mask_id_prob"], (0, 3, 1, 2)), labels["masks"])

  # Get similarity matrix and matching.
  # padded mask[b, j, i] = -1 << other scores, if i >= num_instance[b]
  similarity = cls_score * mask_score
  padded_mask = tf.cast(tf.reshape(tf.range(n), (1, 1, n)), tf.float32)
  padded_mask = tf.cast(
      tf.math.greater_equal(padded_mask,
                            tf.reshape(labels["num_instance"], (b, 1, 1))),
      tf.float32)
  # The constant value for padding has no effect.
  masked_similarity = similarity * (1. - padded_mask) + padded_mask * (-1.)
  matched_mask = matchers_ops.hungarian_matching(-masked_similarity)
  matched_mask = tf.cast(matched_mask, tf.float32) * (1 - padded_mask)
  outputs["matched_mask"] = matched_mask
  # Pos loss
  loss_pos = (
      tf.stop_gradient(cls_score) * (-mask_score) +
      tf.stop_gradient(mask_score) * (-tf.math.log(cls_score)))
  loss_pos = tf.reduce_sum(loss_pos * matched_mask, axis=[1, 2])  # (B,)
  # Neg loss
  matched_pred = tf.cast(tf.reduce_sum(matched_mask, axis=2) > 0,
                         tf.float32)  # (B, N)
  # 0 for void class
  log_loss = -tf.nn.log_softmax(outputs["cls_logits"])[:, :, 0]  # (B, N)
  loss_neg = tf.reduce_sum(log_loss * (1. - matched_pred), axis=-1)  # (B,)

  loss_pq = (alpha * loss_pos + (1 - alpha) * loss_neg) / n
  loss_pq = tf.reduce_mean(loss_pq)
  loss_dict["loss_pq"] = loss_pq


@gin.configurable
def _instance_discrimination_loss(loss_dict: Dict[str, Any],
                                  labels: Dict[str, Any],
                                  outputs: Dict[str, Any],
                                  tau: float = gin.REQUIRED):
  """Instance discrimination loss.

  This method adds the ID loss term to loss_dict directly.

  Args:
    loss_dict: A dictionary for the loss. The values are loss scalars.
    labels: The label dictionary.
    outputs: The output dictionary.
    tau: The temperature term in the loss
  """
  # The normalized feature, shape=(B, H/4, W/4, D)
  g = outputs["max_deep_lab"]["pixel_space_normalized_feature"]
  b, h, w = utilities.resolve_shape(g)[:3]
  # The ground-truth masks, shape=(B, N, H, W) --> (B, N, H/4, W/4)
  m = labels["masks"]
  m = tf.image.resize(
      tf.transpose(m, (0, 2, 3, 1)), (h, w),
      tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  m = tf.transpose(m, (0, 3, 1, 2))
  # The number of ground-truth instance (K), shape=(B,)
  num = labels["num_instance"]
  n = utilities.resolve_shape(m)[1]  # max number of predictions
  # is_void[b, i] = 1 if instance i in batch b is a padded slot.
  is_void = tf.cast(tf.expand_dims(tf.range(n), 0), tf.float32)  # (1, n)
  is_void = tf.cast(
      tf.math.greater_equal(is_void, tf.expand_dims(num, 1)), tf.float32)

  # (B, N, D)
  t = tf.math.l2_normalize(tf.einsum("bhwd,bnhw->bnd", g, m), axis=-1)
  inst_dist_logits = tf.einsum("bhwd,bid->bhwi", g, t) / tau  # (B, H, W, N)
  inst_dist_logits = inst_dist_logits - 100. * tf.reshape(is_void, (b, 1, 1, n))
  mask_id = tf.cast(
      tf.einsum("bnhw,n->bhw", m, tf.range(n, dtype=tf.float32)), tf.int32)
  loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=mask_id, logits=inst_dist_logits)  # B, H, W
  valid_mask = tf.reduce_sum(m, axis=1)
  loss_inst_dist = (
      (tf.reduce_sum(loss_map * valid_mask, axis=[1, 2]) + EPSILON) /
      (tf.reduce_sum(valid_mask, axis=[1, 2]) + EPSILON))
  loss_dict["loss_inst_dist"] = tf.reduce_mean(loss_inst_dist)


@gin.configurable
def _paragraph_grouping_loss(
    loss_dict: Dict[str, Any],
    labels: Dict[str, Any],
    outputs: Dict[str, Any],
    tau: float = gin.REQUIRED,
    loss_mode="vanilla",
    fl_alpha: float = 0.25,
    fl_gamma: float = 2.,
):
  """Instance discrimination loss.

  This method adds the para discrimination loss term to loss_dict directly.

  Args:
    loss_dict: A dictionary for the loss. The values are loss scalars.
    labels: The label dictionary.
    outputs: The output dictionary.
    tau: The temperature term in the loss
    loss_mode: The type of loss.
    fl_alpha: alpha value in focal loss
    fl_gamma: gamma value in focal loss
  """
  if "paragraph_labels" not in labels:
    loss_dict["loss_para"] = 0.
    return
  # step 1:
  # obtain the paragraph labels for each prediction
  # (batch, pred, gt)
  matched_matrix = outputs["instance_output"]["matched_mask"]  # B, N, N
  para_label_gt = labels["paragraph_labels"]["paragraph_ids"]  # B, N
  has_para_label_gt = (
      labels["paragraph_labels"]["has_para_ids"][:, tf.newaxis, tf.newaxis])
  # '0' means no paragraph labels
  pred_label_gt = tf.einsum("bij,bj->bi", matched_matrix,
                            tf.cast(para_label_gt + 1, tf.float32))
  pred_label_gt_pad_col = tf.expand_dims(pred_label_gt, -1)  # b,n,1
  pred_label_gt_pad_row = tf.expand_dims(pred_label_gt, 1)  # b,1,n
  gt_affinity = tf.cast(
      tf.equal(pred_label_gt_pad_col, pred_label_gt_pad_row), tf.float32)
  gt_affinity_mask = (
      has_para_label_gt * pred_label_gt_pad_col * pred_label_gt_pad_row)
  gt_affinity_mask = tf.cast(tf.not_equal(gt_affinity_mask, 0.), tf.float32)

  # step 2:
  # get affinity matrix
  affinity = outputs["para_affinity"]

  # step 3:
  # compute loss
  loss_fn = tf_keras.losses.BinaryCrossentropy(
      from_logits=True,
      label_smoothing=0,
      axis=-1,
      reduction=tf_keras.losses.Reduction.NONE,
      name="para_dist")
  affinity = tf.reshape(affinity, (-1, 1))  # (b*n*n, 1)
  gt_affinity = tf.reshape(gt_affinity, (-1, 1))  # (b*n*n, 1)
  gt_affinity_mask = tf.reshape(gt_affinity_mask, (-1,))  # (b*n*n,)
  pointwise_loss = loss_fn(gt_affinity, affinity / tau)  # (b*n*n,)

  if loss_mode == "vanilla":
    loss = (
        tf.reduce_sum(pointwise_loss * gt_affinity_mask) /
        (tf.reduce_sum(gt_affinity_mask) + EPSILON))
  elif loss_mode == "balanced":
    # pos
    pos_mask = gt_affinity_mask * gt_affinity[:, 0]
    pos_loss = (
        tf.reduce_sum(pointwise_loss * pos_mask) /
        (tf.reduce_sum(pos_mask) + EPSILON))
    # neg
    neg_mask = gt_affinity_mask * (1. - gt_affinity[:, 0])
    neg_loss = (
        tf.reduce_sum(pointwise_loss * neg_mask) /
        (tf.reduce_sum(neg_mask) + EPSILON))
    loss = 0.25 * pos_loss + 0.75 * neg_loss
  elif loss_mode == "focal":
    alpha_wt = fl_alpha * gt_affinity + (1. - fl_alpha) * (1. - gt_affinity)
    prob_pos = tf.math.sigmoid(affinity / tau)
    pt = prob_pos * gt_affinity + (1. - prob_pos) * (1. - gt_affinity)
    fl_loss_pw = tf.stop_gradient(
        alpha_wt * tf.pow(1. - pt, fl_gamma))[:, 0] * pointwise_loss
    loss = (
        tf.reduce_sum(fl_loss_pw * gt_affinity_mask) /
        (tf.reduce_sum(gt_affinity_mask) + EPSILON))
  else:
    raise ValueError(f"Not supported loss mode: {loss_mode}")

  loss_dict["loss_para"] = loss


def _mask_id_xent_loss(loss_dict: Dict[str, Any], labels: Dict[str, Any],
                       outputs: Dict[str, Any]):
  """Mask ID loss.

  This method adds the mask ID loss term to loss_dict directly.

  Args:
    loss_dict: A dictionary for the loss. The values are loss scalars.
    labels: The label dictionary.
    outputs: The output dictionary.
  """
  # (B, N, H, W)
  mask_gt = labels["masks"]
  # B, H, W, N
  mask_id_logits = outputs["instance_output"]["mask_id_logits"]
  # B, N, N
  matched_matrix = outputs["instance_output"]["matched_mask"]
  # B, N
  gt_to_pred_id = tf.cast(tf.math.argmax(matched_matrix, axis=1), tf.float32)
  # B, H, W
  mask_id_labels = tf.cast(
      tf.einsum("bnhw,bn->bhw", mask_gt, gt_to_pred_id), tf.int32)
  loss_map = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=mask_id_labels, logits=mask_id_logits)
  valid_mask = tf.reduce_sum(mask_gt, axis=1)
  loss_mask_id = (
      (tf.reduce_sum(loss_map * valid_mask, axis=[1, 2]) + EPSILON) /
      (tf.reduce_sum(valid_mask, axis=[1, 2]) + EPSILON))
  loss_dict["loss_mask_id"] = tf.reduce_mean(loss_mask_id)
