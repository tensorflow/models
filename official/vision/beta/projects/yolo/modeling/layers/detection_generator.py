"""Contains common building blocks for yolo neural networks."""
import tensorflow as tf
import tensorflow.keras as ks
import tensorflow.keras.backend as K

# from official.vision.beta.projects.yolo.ops import loss_utils
from official.vision.beta.projects.yolo.ops import box_ops
# from official.vision.beta.projects.yolo.losses import yolo_loss
# from official.vision.beta.projects.yolo.ops import nms_ops


@ks.utils.register_keras_serializable(package='yolo')
class YoloLayer(ks.Model):

  def __init__(self,
               masks,
               anchors,
               classes,
               iou_thresh=0.0,
               ignore_thresh=0.7,
               truth_thresh=1.0,
               nms_thresh=0.6,
               max_delta=10.0,
               loss_type='ciou',
               use_tie_breaker=True,
               iou_normalizer=1.0,
               cls_normalizer=1.0,
               obj_normalizer=1.0,
               use_scaled_loss=False,
               darknet = None,
               pre_nms_points=5000,
               label_smoothing=0.0,
               max_boxes=200,
               new_cords=False,
               path_scale=None,
               scale_xy=None,
               nms_type='greedy',
               objectness_smooth=False,
               **kwargs):
    """
    parameters for the loss functions used at each detection head output

scale_anchors: `int` for how much to scale this level to get the orginal
  input shape

    Args:
      masks: `List[int]` for the output level that this specific model output
        level.
      anchors: `List[List[int]]` for the anchor boxes that are used in the
        model.
      classes: `int` for the number of classes.
      iou_thresh: `float` to use many anchors per object if IoU(Obj, Anchor) >
        iou_thresh.
      ignore_thresh: `float` for the IOU value over which the loss is not
        propagated, and a detection is assumed to have been made.
      truth_thresh: `float` for the IOU value over which the loss is propagated
        despite a detection being made'.
      nms_thresh: `float` for the minimum IOU value for an overlap.
      max_delta: gradient clipping to apply to the box loss.
      loss_type: `str` for the typeof iou loss to use with in {ciou, diou,
        giou, iou}.

      iou_normalizer: `float` for how much to scale the loss on the IOU or the
        boxes.
      cls_normalizer: `float` for how much to scale the loss on the classes.
      obj_normalizer: `float` for how much to scale loss on the detection map.
      objectness_smooth: `float` for how much to smooth the loss on the
        detection map.
      use_scaled_loss: `bool` for whether to use the scaled loss
        or the traditional loss.

      label_smoothing: `float` for how much to smooth the loss on the classes.
      new_cords: `bool` for which scaling type to use.
      scale_xy: dictionary `float` values inidcating how far each pixel can see
        outside of its containment of 1.0. a value of 1.2 indicates there is a
        20% extended radius around each pixel that this specific pixel can
        predict values for a center at. the center can range from 0 - value/2
        to 1 + value/2, this value is set in the yolo filter, and resused here.
        there should be one value for scale_xy for each level from min_level to
        max_level.
      nms_type: "greedy",
      nms_thresh: 0.6,
      iou_thresh: 0.213,
      name=None,


    Return:
      loss: `float` for the actual loss.
      box_loss: `float` loss on the boxes used for metrics.
      conf_loss: `float` loss on the confidence used for metrics.
      class_loss: `float` loss on the classes used for metrics.
      avg_iou: `float` metric for the average iou between predictions
        and ground truth.
      avg_obj: `float` metric for the average confidence of the model
        for predictions.
      recall50: `float` metric for how accurate the model is.
      precision50: `float` metric for how precise the model is.
    """
    super().__init__(**kwargs)
    self._masks = masks
    self._anchors = anchors
    self._thresh = iou_thresh
    self._ignore_thresh = ignore_thresh
    self._truth_thresh = truth_thresh
    self._iou_normalizer = iou_normalizer
    self._cls_normalizer = cls_normalizer
    self._obj_normalizer = obj_normalizer
    self._objectness_smooth = objectness_smooth
    self._nms_thresh = nms_thresh
    self._max_boxes = max_boxes
    self._max_delta = max_delta
    self._classes = classes
    self._loss_type = loss_type
    self._use_tie_breaker = use_tie_breaker

    self._use_scaled_loss = use_scaled_loss
    self._darknet = darknet

    self._pre_nms_points = pre_nms_points
    self._label_smoothing = label_smoothing
    self._keys = list(masks.keys())
    self._len_keys = len(self._keys)
    self._new_cords = new_cords
    self._path_scale = path_scale or {
        key: 2**int(key) for key, _ in masks.items()
    }

    self._nms_types = {
        'greedy': 1,
        'iou': 2,
        'giou': 3,
        'ciou': 4,
        'diou': 5,
        'class_independent': 6,
        'weighted_diou': 7
    }

    self._nms_type = self._nms_types[nms_type]

    if self._nms_type >= 2 and self._nms_type <= 5:
      self._nms = nms_ops.TiledNMS(iou_type=nms_type)

    self._scale_xy = scale_xy or {key: 1.0 for key, _ in masks.items()}

    self._generator = {}
    self._len_mask = {}
    for key in self._keys:
      anchors = [self._anchors[mask] for mask in self._masks[key]]
      self._generator[key] = self.get_generators(anchors, self._path_scale[key],
                                                 key)
      self._len_mask[key] = len(self._masks[key])
    return

  def get_generators(self, anchors, path_scale, path_key):
    # anchor_generator = loss_utils.GridGenerator(
    #     anchors, scale_anchors=path_scale)
    # return anchor_generator
    return None

  def rm_nan_inf(self, x, val=0.0):
    x = tf.where(tf.math.is_nan(x), tf.cast(val, dtype=x.dtype), x)
    x = tf.where(tf.math.is_inf(x), tf.cast(val, dtype=x.dtype), x)
    return x

  def parse_prediction_path(self, key, inputs):
    shape_ = tf.shape(inputs)
    shape = inputs.get_shape().as_list()
    batchsize, height, width = shape_[0], shape[1], shape[2]

    generator = self._generator[key]
    len_mask = self._len_mask[key]
    scale_xy = self._scale_xy[key]

    # reshape the yolo output to (batchsize,
    #                             width,
    #                             height,
    #                             number_anchors,
    #                             remaining_points)

    data = tf.reshape(inputs, [-1, height, width, len_mask, self._classes + 5])

    # use the grid generator to get the formatted anchor boxes and grid points
    # in shape [1, height, width, 2]
    centers, anchors = generator(height, width, batchsize, dtype=data.dtype)

    # # tempcode
    # centers /= tf.cast([width, height], centers.dtype)
    # anchors /= tf.cast([width, height], anchors.dtype)

    # split the yolo detections into boxes, object score map, classes
    boxes, obns_scores, class_scores = tf.split(
        data, [4, 1, self._classes], axis=-1)

    # determine the number of classes
    classes = class_scores.get_shape().as_list()[
        -1]  #tf.shape(class_scores)[-1]

    # # configurable to use the new coordinates in scaled Yolo v4 or not
    # if not self._new_cords[key]:
    #   # coordinates from scaled yolov4
    #   _, _, boxes = yolo_loss.get_predicted_box(
    #       tf.cast(height, data.dtype), tf.cast(width, data.dtype), boxes,
    #       anchors, centers, scale_xy)
    # else:
    #   # coordinates from regular yolov3 - v4
    #   _, _, boxes = yolo_loss.get_predicted_box_newcords(
    #       tf.cast(height, data.dtype), tf.cast(width, data.dtype), boxes,
    #       anchors, centers, scale_xy)
    boxes = None

    # convert boxes from yolo(x, y, w. h) to tensorflow(ymin, xmin, ymax, xmax)
    boxes = box_ops.xcycwh_to_yxyx(boxes)

    # activate and detection map
    obns_scores = tf.math.sigmoid(obns_scores)

    # threshold the detection map
    obns_mask = tf.cast(obns_scores > self._thresh, obns_scores.dtype)

    # convert detection map to class detection probabailities
    class_scores = tf.math.sigmoid(class_scores) * obns_mask * obns_scores
    class_scores *= tf.cast(class_scores > self._thresh, class_scores.dtype)

    fill = height * width * len_mask
    # platten predictions to [batchsize, N, -1] for non max supression
    boxes = tf.reshape(boxes, [-1, fill, 4])
    class_scores = tf.reshape(class_scores, [-1, fill, classes])
    obns_scores = tf.reshape(obns_scores, [-1, fill])

    return obns_scores, boxes, class_scores

  def call(self, inputs):
    boxes = []
    class_scores = []
    object_scores = []
    levels = list(inputs.keys())
    min_level = int(min(levels))
    max_level = int(max(levels))

    # aggregare boxes over each scale
    for i in range(min_level, max_level + 1):
      key = str(i)
      object_scores_, boxes_, class_scores_ = self.parse_prediction_path(
          key, inputs[key])
      boxes.append(boxes_)
      class_scores.append(class_scores_)
      object_scores.append(object_scores_)

    # colate all predicitons
    boxes = tf.concat(boxes, axis=1)
    object_scores = K.concatenate(object_scores, axis=1)
    class_scores = K.concatenate(class_scores, axis=1)

    # # apply nms
    # if self._nms_type == 7:
    #   boxes, class_scores, object_scores = nms_ops.non_max_suppression2(
    #       boxes,
    #       class_scores,
    #       object_scores,
    #       self._max_boxes,
    #       pre_nms_thresh = self._thresh,
    #       nms_thresh = self._nms_thresh,
    #       prenms_top_k=self._pre_nms_points)
    # elif self._nms_type == 6:
    #   boxes, class_scores, object_scores = nms_ops.nms(
    #       boxes,
    #       class_scores,
    #       object_scores,
    #       self._max_boxes,
    #       self._thresh,
    #       self._nms_thresh,
    #       prenms_top_k=self._pre_nms_points)
    # elif self._nms_type == 1:
    #   # greedy NMS
    #   boxes = tf.cast(boxes, dtype=tf.float32)
    #   class_scores = tf.cast(class_scores, dtype=tf.float32)
    #   nms_items = tf.image.combined_non_max_suppression(
    #       tf.expand_dims(boxes, axis=-2),
    #       class_scores,
    #       self._pre_nms_points,
    #       self._max_boxes,
    #       iou_threshold=self._nms_thresh,
    #       score_threshold=self._thresh)
    #   # cast the boxes and predicitons abck to original datatype
    #   boxes = tf.cast(nms_items.nmsed_boxes, object_scores.dtype)
    #   class_scores = tf.cast(nms_items.nmsed_classes, object_scores.dtype)
    #   object_scores = tf.cast(nms_items.nmsed_scores, object_scores.dtype)
    #
    # else:
    #   boxes = tf.cast(boxes, dtype=tf.float32)
    #   class_scores = tf.cast(class_scores, dtype=tf.float32)
    #   boxes, confidence, classes, valid = self._nms.complete_nms(
    #       tf.expand_dims(boxes, axis=-2),
    #       class_scores,
    #       pre_nms_top_k=self._pre_nms_points,
    #       max_num_detections=self._max_boxes,
    #       nms_iou_threshold=self._nms_thresh,
    #       pre_nms_score_threshold=self._thresh)
    #   boxes = tf.cast(boxes, object_scores.dtype)
    #   class_scores = tf.cast(classes, object_scores.dtype)
    #   object_scores = tf.cast(confidence, object_scores.dtype)

    # compute the number of valid detections
    num_detections = tf.math.reduce_sum(tf.math.ceil(object_scores), axis=-1)

    # format and return
    return {
        'bbox': boxes,
        'classes': class_scores,
        'confidence': object_scores,
        'num_detections': num_detections,
    }

  @property
  def losses(self):
    """ Generates a dictionary of losses to apply to each path

    Done in the detection generator because all parameters are the same
    across both loss and detection generator
    """
    # loss_dict = {}
    # for key in self._keys:
    #   loss_dict[key] = yolo_loss.Yolo_Loss(
    #       classes=self._classes,
    #       anchors=self._anchors,
    #       darknet=self._darknet,
    #       truth_thresh=self._truth_thresh[key],
    #       ignore_thresh=self._ignore_thresh[key],
    #       loss_type=self._loss_type[key],
    #       iou_normalizer=self._iou_normalizer[key],
    #       cls_normalizer=self._cls_normalizer[key],
    #       obj_normalizer=self._obj_normalizer[key],
    #       new_cords=self._new_cords[key],
    #       objectness_smooth=self._objectness_smooth[key],
    #       use_scaled_loss=self._use_scaled_loss,
    #       label_smoothing=self._label_smoothing,
    #       mask=self._masks[key],
    #       max_delta=self._max_delta[key],
    #       scale_anchors=self._path_scale[key],
    #       scale_x_y=self._scale_xy[key])
    # return loss_dict
    return None

  def get_config(self):
    return {
        'masks': dict(self._masks),
        'anchors': [list(a) for a in self._anchors],
        'thresh': self._thresh,
        'max_boxes': self._max_boxes,
    }
