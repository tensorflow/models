import tensorflow as tf
from official.vision.beta.ops import box_ops as box_utils
from official.vision.beta.projects.yolo.ops import box_ops as box_ops

NMS_TILE_SIZE = 512

class TiledNMS():
  IOU_TYPES = {'diou': 0, 'giou': 1, 'ciou': 2, 'iou': 3}

  def __init__(self, iou_type='diou', beta=0.6):
    '''Initialization for all non max suppression operations mainly used to
    select hyperparameters for the iou type and scaling.

    Args:
      iou_type: `str` for the version of IOU to use {diou, giou, ciou, iou}.
      beta: `float` for the amount to scale regularization on distance iou.
    '''
    self._iou_type = TiledNMS.IOU_TYPES[iou_type]
    self._beta = beta

  def _self_suppression(self, iou, _, iou_sum):
    batch_size = tf.shape(iou)[0]
    can_suppress_others = tf.cast(
        tf.reshape(tf.reduce_max(iou, 1) <= 0.5, [batch_size, -1, 1]),
        iou.dtype)
    iou_suppressed = tf.reshape(
        tf.cast(tf.reduce_max(can_suppress_others * iou, 1) <= 0.5, iou.dtype),
        [batch_size, -1, 1]) * iou
    iou_sum_new = tf.reduce_sum(iou_suppressed, [1, 2])
    return [
        iou_suppressed,
        tf.reduce_any(iou_sum - iou_sum_new > 0.5), iou_sum_new
    ]

  def _cross_suppression(self, boxes, box_slice, iou_threshold, inner_idx):
    batch_size = tf.shape(boxes)[0]
    new_slice = tf.slice(boxes, [0, inner_idx * NMS_TILE_SIZE, 0],
                         [batch_size, NMS_TILE_SIZE, 4])
    #iou = box_ops.bbox_overlap(new_slice, box_slice)
    iou = box_ops.aggregated_comparitive_iou(
        new_slice, box_slice, beta=self._beta, iou_type=self._iou_type)
    ret_slice = tf.expand_dims(
        tf.cast(tf.reduce_all(iou < iou_threshold, [1]), box_slice.dtype),
        2) * box_slice
    return boxes, ret_slice, iou_threshold, inner_idx + 1

  def _suppression_loop_body(self, boxes, iou_threshold, output_size, idx):
    """Process boxes in the range [idx*NMS_TILE_SIZE, (idx+1)*NMS_TILE_SIZE).

    Args:
      boxes: a tensor with a shape of [batch_size, anchors, 4].
      iou_threshold: a float representing the threshold for whether boxes
        overlap too much with respect to IOU.
      output_size: an int32 tensor of size [batch_size]. Representing the number
        of selected boxes for each batch.
      idx: an integer scalar representing an induction variable.

    Returns:
      boxes: updated boxes.
      iou_threshold: pass down iou_threshold to the next iteration.
      output_size: the updated output_size.
      idx: the updated induction variable.
    """
    num_tiles = tf.shape(boxes)[1] // NMS_TILE_SIZE
    batch_size = tf.shape(boxes)[0]

    # Iterates over tiles that can possibly suppress the current tile.
    box_slice = tf.slice(boxes, [0, idx * NMS_TILE_SIZE, 0],
                         [batch_size, NMS_TILE_SIZE, 4])
    _, box_slice, _, _ = tf.while_loop(
        lambda _boxes, _box_slice, _threshold, inner_idx: inner_idx < idx,
        self._cross_suppression,
        [boxes, box_slice, iou_threshold,
         tf.constant(0)])

    # Iterates over the current tile to compute self-suppression.
    # iou = box_ops.bbox_overlap(box_slice, box_slice)
    iou = box_ops.aggregated_comparitive_iou(
        box_slice, box_slice, beta=self._beta, iou_type=self._iou_type)
    mask = tf.expand_dims(
        tf.reshape(tf.range(NMS_TILE_SIZE), [1, -1]) > tf.reshape(
            tf.range(NMS_TILE_SIZE), [-1, 1]), 0)
    iou *= tf.cast(tf.logical_and(mask, iou >= iou_threshold), iou.dtype)
    suppressed_iou, _, _ = tf.while_loop(
        lambda _iou, loop_condition, _iou_sum: loop_condition,
        self._self_suppression,
        [iou, tf.constant(True),
         tf.reduce_sum(iou, [1, 2])])
    suppressed_box = tf.reduce_sum(suppressed_iou, 1) > 0
    box_slice *= tf.expand_dims(1.0 - tf.cast(suppressed_box, box_slice.dtype),
                                2)

    # Uses box_slice to update the input boxes.
    mask = tf.reshape(
        tf.cast(tf.equal(tf.range(num_tiles), idx), boxes.dtype), [1, -1, 1, 1])
    boxes = tf.tile(tf.expand_dims(
        box_slice, [1]), [1, num_tiles, 1, 1]) * mask + tf.reshape(
            boxes, [batch_size, num_tiles, NMS_TILE_SIZE, 4]) * (1 - mask)
    boxes = tf.reshape(boxes, [batch_size, -1, 4])

    # Updates output_size.
    output_size += tf.reduce_sum(
        tf.cast(tf.reduce_any(box_slice > 0, [2]), tf.int32), [1])
    return boxes, iou_threshold, output_size, idx + 1

  def _sorted_non_max_suppression_padded(self, scores, boxes, max_output_size,
                                         iou_threshold):
    """A wrapper that handles non-maximum suppression.

    Assumption:
      * The boxes are sorted by scores unless the box is a dot (all coordinates
        are zero).
      * Boxes with higher scores can be used to suppress boxes with lower
        scores.

    The overall design of the algorithm is to handle boxes tile-by-tile:

    boxes = boxes.pad_to_multiply_of(tile_size)
    num_tiles = len(boxes) // tile_size
    output_boxes = []
    for i in range(num_tiles):
      box_tile = boxes[i*tile_size : (i+1)*tile_size]
      for j in range(i - 1):
        suppressing_tile = boxes[j*tile_size : (j+1)*tile_size]
        iou = bbox_overlap(box_tile, suppressing_tile)
        # if the box is suppressed in iou, clear it to a dot
        box_tile *= _update_boxes(iou)
      # Iteratively handle the diagonal tile.
      iou = _box_overlap(box_tile, box_tile)
      iou_changed = True
      while iou_changed:
        # boxes that are not suppressed by anything else
        suppressing_boxes = _get_suppressing_boxes(iou)
        # boxes that are suppressed by suppressing_boxes
        suppressed_boxes = _get_suppressed_boxes(iou, suppressing_boxes)
        # clear iou to 0 for boxes that are suppressed, as they cannot be used
        # to suppress other boxes any more
        new_iou = _clear_iou(iou, suppressed_boxes)
        iou_changed = (new_iou != iou)
        iou = new_iou
      # remaining boxes that can still suppress others, are selected boxes.
      output_boxes.append(_get_suppressing_boxes(iou))
      if len(output_boxes) >= max_output_size:
        break

    Args:
      scores: a tensor with a shape of [batch_size, anchors].
      boxes: a tensor with a shape of [batch_size, anchors, 4].
      max_output_size: a scalar integer `Tensor` representing the maximum number
        of boxes to be selected by non max suppression.
      iou_threshold: a float representing the threshold for whether boxes
        overlap too much with respect to IOU.

    Returns:
      nms_scores: a tensor with a shape of [batch_size, anchors]. It has same
        dtype as input scores.
      nms_proposals: a tensor with a shape of [batch_size, anchors, 4]. It has
        same dtype as input boxes.
    """
    batch_size = tf.shape(boxes)[0]
    num_boxes = tf.shape(boxes)[1]
    pad = tf.cast(
        tf.math.ceil(tf.cast(num_boxes, tf.float32) / NMS_TILE_SIZE),
        tf.int32) * NMS_TILE_SIZE - num_boxes
    boxes = tf.pad(tf.cast(boxes, tf.float32), [[0, 0], [0, pad], [0, 0]])
    scores = tf.pad(
        tf.cast(scores, tf.float32), [[0, 0], [0, pad]], constant_values=-1)
    num_boxes += pad

    def _loop_cond(unused_boxes, unused_threshold, output_size, idx):
      return tf.logical_and(
          tf.reduce_min(output_size) < max_output_size,
          idx < num_boxes // NMS_TILE_SIZE)

    selected_boxes, _, output_size, _ = tf.while_loop(
        _loop_cond, self._suppression_loop_body, [
            boxes, iou_threshold,
            tf.zeros([batch_size], tf.int32),
            tf.constant(0)
        ])
    idx = num_boxes - tf.cast(
        tf.nn.top_k(
            tf.cast(tf.reduce_any(selected_boxes > 0, [2]), tf.int32) *
            tf.expand_dims(tf.range(num_boxes, 0, -1), 0), max_output_size)[0],
        tf.int32)
    idx = tf.minimum(idx, num_boxes - 1)
    idx = tf.reshape(
        idx + tf.reshape(tf.range(batch_size) * num_boxes, [-1, 1]), [-1])
    boxes = tf.reshape(
        tf.gather(tf.reshape(boxes, [-1, 4]), idx),
        [batch_size, max_output_size, 4])
    boxes = boxes * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1, 1]) < tf.reshape(
            output_size, [-1, 1, 1]), boxes.dtype)
    scores = tf.reshape(
        tf.gather(tf.reshape(scores, [-1, 1]), idx),
        [batch_size, max_output_size])
    scores = scores * tf.cast(
        tf.reshape(tf.range(max_output_size), [1, -1]) < tf.reshape(
            output_size, [-1, 1]), scores.dtype)

    return scores, boxes

  def _select_top_k_scores(self, scores_in, pre_nms_num_detections):
    # batch_size, num_anchors, num_class = scores_in.get_shape().as_list()
    scores_shape = scores_in.get_shape().as_list()  #tf.shape(scores_in)
    batch_size, num_anchors, num_class = scores_shape[0], scores_shape[
        1], scores_shape[2]
    scores_trans = tf.transpose(scores_in, perm=[0, 2, 1])
    scores_trans = tf.reshape(scores_trans, [-1, num_anchors])

    top_k_scores, top_k_indices = tf.nn.top_k(
        scores_trans, k=pre_nms_num_detections, sorted=True)

    top_k_scores = tf.reshape(top_k_scores,
                              [-1, num_class, pre_nms_num_detections])
    top_k_indices = tf.reshape(top_k_indices,
                               [-1, num_class, pre_nms_num_detections])

    return tf.transpose(top_k_scores,
                        [0, 2, 1]), tf.transpose(top_k_indices, [0, 2, 1])

  def complete_nms(self,
                   boxes,
                   scores,
                   pre_nms_top_k=5000,
                   pre_nms_score_threshold=0.05,
                   nms_iou_threshold=0.5,
                   max_num_detections=100):
    """Generate the final detections given the model outputs.

    This implementation unrolls classes dimension while using the tf.while_loop
    to implement the batched NMS, so that it can be parallelized at the batch
    dimension. It should give better performance compared to v1 implementation.
    It is TPU compatible.

    Args:
      boxes: a tensor with shape [batch_size, N, num_classes, 4] or [batch_size,
        N, 1, 4], which box predictions on all feature levels. The N is the
        number of total anchors on all levels.
      scores: a tensor with shape [batch_size, N, num_classes], which stacks
        class probability on all feature levels. The N is the number of total
        anchors on all levels. The num_classes is the number of classes the
        model predicted. Note that the class_outputs here is the raw score.
      pre_nms_top_k: an int number of top candidate detections per class
        before NMS.
      pre_nms_score_threshold: a float representing the threshold for deciding
        when to remove boxes based on score.
      nms_iou_threshold: a float representing the threshold for deciding whether
        boxes overlap too much with respect to IOU.
      max_num_detections: a scalar representing maximum number of boxes retained
        over all classes.

    Returns:
      nms_boxes: `float` Tensor of shape [batch_size, max_num_detections, 4]
        representing top detected boxes in [y1, x1, y2, x2].
      nms_scores: `float` Tensor of shape [batch_size, max_num_detections]
        representing sorted confidence scores for detected boxes. The values are
        between [0, 1].
      nms_classes: `int` Tensor of shape [batch_size, max_num_detections]
        representing classes for detected boxes.
      valid_detections: `int` Tensor of shape [batch_size] only the top
        `valid_detections` boxes are valid detections.
    """
    with tf.name_scope('nms'):
      nmsed_boxes = []
      nmsed_classes = []
      nmsed_scores = []
      valid_detections = []
      boxes_shape = boxes.get_shape().as_list()
      batch_size, _, num_classes_for_box, _ = (boxes_shape[0], boxes_shape[1],
                                               boxes_shape[2], boxes_shape[3])

      scores_shape = scores.get_shape().as_list()
      _, total_anchors, num_classes = (scores_shape[0], scores_shape[1],
                                       scores_shape[2])

      scores, indices = self._select_top_k_scores(
          scores, tf.math.minimum(total_anchors, pre_nms_top_k))

      for i in range(num_classes):
        boxes_i = boxes[:, :, min(num_classes_for_box - 1, i), :]
        scores_i = scores[:, :, i]
        # Obtains pre_nms_top_k before running NMS.
        boxes_i = tf.gather(boxes_i, indices[:, :, i], batch_dims=1, axis=1)

        # Filter out scores.
        boxes_i, scores_i = box_utils.filter_boxes_by_scores(
            boxes_i, scores_i, min_score_threshold=pre_nms_score_threshold)

        (nmsed_scores_i,
         nmsed_boxes_i) = self._sorted_non_max_suppression_padded(
             tf.cast(scores_i, tf.float32),
             tf.cast(boxes_i, tf.float32),
             max_num_detections,
             iou_threshold=nms_iou_threshold)
        nmsed_classes_i = tf.ones_like(nmsed_scores_i, dtype=tf.int32) * i

        #tf.fill([batch_size, max_num_detections], i)
        nmsed_boxes.append(nmsed_boxes_i)
        nmsed_scores.append(nmsed_scores_i)
        nmsed_classes.append(nmsed_classes_i)

    nmsed_boxes = tf.concat(nmsed_boxes, axis=1)
    nmsed_scores = tf.concat(nmsed_scores, axis=1)
    nmsed_classes = tf.concat(nmsed_classes, axis=1)
    nmsed_scores, indices = tf.nn.top_k(
        nmsed_scores, k=max_num_detections, sorted=True)
    nmsed_boxes = tf.gather(nmsed_boxes, indices, batch_dims=1, axis=1)
    nmsed_classes = tf.gather(nmsed_classes, indices, batch_dims=1)
    valid_detections = tf.reduce_sum(
        input_tensor=tf.cast(tf.greater(nmsed_scores, -1), tf.int32), axis=1)

    return nmsed_boxes, nmsed_scores, nmsed_classes, valid_detections


BASE_NMS = TiledNMS(iou_type='iou', beta=0.6)


def sorted_non_max_suppression_padded(scores, boxes, max_output_size,
                                      iou_threshold):
  """wrapper function to match NMS found in official/vision/beta/ops/nms.py"""
  return BASE_NMS._sorted_non_max_suppression_padded(scores, boxes,
                                                     max_output_size,
                                                     iou_threshold)


def sort_drop(objectness, box, classificationsi, k):
  """This function sorts and drops boxes such that there are only k boxes
  sorted by number the objectness or confidence

  Args:
    objectness: a `Tensor` of shape [batch size, N] that needs to be
      filtered.
    box: a `Tensor` of shape [batch size, N, 4] that needs to be filtered.
    classificationsi: a `Tensor` of shape [batch size, N, num_classes] that
      needs to be filtered.
    k: a `integer` for the maximum number of boxes to keep after filtering

  Return:
    objectness: filtered `Tensor` of shape [batch size, k]
    boxes: filtered `Tensor` of shape [batch size, k, 4]
    classifications: filtered `Tensor` of shape [batch size, k, num_classes]
  """
  # find rhe indexes for the boxes based on the scores
  objectness, ind = tf.math.top_k(objectness, k=k)

  # build the indexes
  ind_m = tf.ones_like(ind) * tf.expand_dims(
      tf.range(0,
               tf.shape(objectness)[0]), axis=-1)
  bind = tf.stack([tf.reshape(ind_m, [-1]), tf.reshape(ind, [-1])], axis=-1)

  # gather all the high confidence boxes and classes
  box = tf.gather_nd(box, bind)
  classifications = tf.gather_nd(classificationsi, bind)

  # resize and clip the boxes
  bsize = tf.shape(ind)[0]
  box = tf.reshape(box, [bsize, k, -1])
  classifications = tf.reshape(classifications, [bsize, k, -1])
  return objectness, box, classifications


def segment_nms(boxes, classes, confidence, k, iou_thresh):
  """This is a quick nms that works on very well for small values of k, this
  was developed to operate for tflite models as the tiled NMS is far too slow
  and typically is not able to compile with tflite. This NMS does not account
  for classes, and only works to quickly filter boxes on phones.

  Args:
    boxes: a `Tensor` of shape [batch size, N, 4] that needs to be filtered.
    classes: a `Tensor` of shape [batch size, N, num_classes] that needs to be
      filtered.
    confidence: a `Tensor` of shape [batch size, N] that needs to be
      filtered.
    k: a `integer` for the maximum number of boxes to keep after filtering
    iou_thresh: a `float` for the value above which boxes are considered to be
      too similar, the closer to 1.0 the less that gets through.

  Return:
    boxes: filtered `Tensor` of shape [batch size, k, 4]
    classes: filtered `Tensor` of shape [batch size, k, num_classes] t
    confidence: filtered `Tensor` of shape [batch size, k]
  """
  mrange = tf.range(k)
  mask_x = tf.tile(
      tf.transpose(tf.expand_dims(mrange, axis=-1), perm=[1, 0]), [k, 1])
  mask_y = tf.tile(tf.expand_dims(mrange, axis=-1), [1, k])
  mask_diag = tf.expand_dims(mask_x > mask_y, axis=0)

  iou = box_ops.aggregated_comparitive_iou(boxes, iou_type=0)

  # duplicate boxes
  iou_mask = iou >= iou_thresh
  iou_mask = tf.logical_and(mask_diag, iou_mask)
  iou *= tf.cast(iou_mask, iou.dtype)

  can_suppress_others = 1 - tf.cast(
      tf.reduce_any(iou_mask, axis=-2), boxes.dtype)

  # build a mask of the boxes that need to exit
  raw = tf.cast(can_suppress_others, boxes.dtype)

  boxes *= tf.expand_dims(raw, axis=-1)
  confidence *= tf.cast(raw, confidence.dtype)
  classes *= tf.cast(tf.expand_dims(raw, axis=-1), classes.dtype)
  return boxes, classes, confidence


def nms(boxes,
        classes,
        confidence,
        k,
        pre_nms_thresh,
        nms_thresh,
        prenms_top_k=500):
  """This is a quick nms that works on very well for small values of k, this
  was developed to operate for tflite models as the tiled NMS is far too slow
  and typically is not able to compile with tflite. This NMS does not account
  for classes, and only works to quickly filter boxes on phones.

  Args:
    boxes: a `Tensor` of shape [batch size, N, 4] that needs to be filtered.
    classes: a `Tensor` of shape [batch size, N, num_classes] that needs to be
      filtered.
    confidence: a `Tensor` of shape [batch size, N] that needs to be
      filtered.
    k: a `integer` for the maximum number of boxes to keep after filtering
    nms_thresh: a `float` for the value above which boxes are considered to be
      too similar, the closer to 1.0 the less that gets through.
    pre_nms_top_k: an int number of top candidate detections per class
      before NMS.

  Return:
    boxes: filtered `Tensor` of shape [batch size, k, 4]
    classes: filtered `Tensor` of shape [batch size, k, num_classes]
    confidence: filtered `Tensor` of shape [batch size, k]
  """

  # sort the boxes
  confidence = tf.reduce_max(classes, axis=-1)
  confidence, boxes, classes = sort_drop(confidence, boxes, classes,
                                         prenms_top_k)

  # apply non max supression
  boxes, classes, confidence = segment_nms(boxes, classes, confidence,
                                           prenms_top_k, nms_thresh)

  # sort the classes of the unspressed boxes
  class_confidence, class_ind = tf.math.top_k(
      classes, k=tf.shape(classes)[-1], sorted=True)

  # set low confidence classes to zero
  mask = tf.fill(
      tf.shape(class_confidence),
      tf.cast(pre_nms_thresh, dtype=class_confidence.dtype))
  mask = tf.math.ceil(tf.nn.relu(class_confidence - mask))
  class_confidence = tf.cast(class_confidence, mask.dtype) * mask
  class_ind = tf.cast(class_ind, mask.dtype) * mask

  # sort the classes and take the top_n as an short cut to doing a true
  # per class NMS
  top_n = tf.math.minimum(100, tf.shape(classes)[-1])
  classes = class_ind[..., :top_n]
  confidence = class_confidence[..., :top_n]

  # reshape and map multiple classes to boxes
  boxes = tf.expand_dims(boxes, axis=-2)
  boxes = tf.tile(boxes, [1, 1, top_n, 1])

  shape = tf.shape(boxes)
  boxes = tf.reshape(boxes, [shape[0], -1, 4])
  classes = tf.reshape(classes, [shape[0], -1])
  confidence = tf.reshape(confidence, [shape[0], -1])

  # drop all the low class confidence boxes again
  confidence, boxes, classes = sort_drop(confidence, boxes, classes, k)

  # mask the boxes classes and scores then toa final reshape before returning
  mask = tf.fill(
      tf.shape(confidence), tf.cast(pre_nms_thresh, dtype=confidence.dtype))
  mask = tf.math.ceil(tf.nn.relu(confidence - mask))
  confidence = confidence * mask
  mask = tf.expand_dims(mask, axis=-1)
  boxes = boxes * mask
  classes = classes * mask

  classes = tf.squeeze(classes, axis=-1)
  return boxes, classes, confidence
