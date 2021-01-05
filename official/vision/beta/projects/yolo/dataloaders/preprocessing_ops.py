import tensorflow as tf
import tensorflow.keras.backend as K
import tensorflow_addons as tfa
from official.vision.beta.projects.yolo.utils.iou_ops import *


def resize_crop_filter(image, boxes, default_width, default_height,
                       target_width, target_height):
  with tf.name_scope("resize_crop_filter"):
    image = tf.image.resize(image, (target_width, target_height))
    image = tf.image.resize_with_crop_or_pad(image,
                                             target_height=default_height,
                                             target_width=default_width)

    default_width = tf.cast(default_width, boxes.dtype)
    default_height = tf.cast(default_height, boxes.dtype)
    target_width = tf.cast(target_width, boxes.dtype)
    target_height = tf.cast(target_height, boxes.dtype)

    aspect_change_width = target_width / default_width
    aspect_change_height = target_height / default_height

    x, y, width, height = tf.split(boxes, 4, axis=-1)
    x = (x - 0.5) * target_width / default_width + 0.5
    y = (y - 0.5) * target_height / default_height + 0.5
    width = width * aspect_change_width
    height = height * aspect_change_height
    boxes = tf.concat([x, y, width, height], axis=-1)
  return image, boxes


def random_translate(image, box, t, seed=10):
  t_x = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
  t_y = tf.random.uniform(minval=-t, maxval=t, shape=(), dtype=tf.float32)
  box = translate_boxes(box, t_x, t_y)
  image = translate_image(image, t_x, t_y)
  return image, box


def translate_boxes(box, translate_x, translate_y):
  with tf.name_scope("translate_boxs"):
    x = box[..., 0] + translate_x
    y = box[..., 1] + translate_y
    box = tf.stack([x, y, box[..., 2], box[..., 3]], axis=-1)
    box.set_shape([None, 4])
  return box


def translate_image(image, translate_x, translate_y):
  with tf.name_scope("translate_image"):
    if (translate_x != 0 and translate_y != 0):
      image_jitter = tf.convert_to_tensor([translate_x, translate_y])
      image_jitter.set_shape([2])
      image = tfa.image.translate(
          image, image_jitter * tf.cast(tf.shape(image)[1], tf.float32))
  return image


def get_best_anchor(y_true, anchors, width=1, height=1):
  """
    get the correct anchor that is assoiciated with each box using IOU betwenn
    input anchors and gt
    Args:
        y_true: tf.Tensor[] for the list of bounding boxes in the yolo format
        anchors: list or tensor for the anchor boxes to be used in prediction
            found via Kmeans
        size: size of the image that the bounding boxes were selected at 416 is
            the default for the original YOLO model
    return:
        tf.Tensor: y_true with the anchor associated with each ground truth box
            known
    """
  with tf.name_scope("get_anchor"):
    width = tf.cast(width, dtype=tf.float32)
    height = tf.cast(height, dtype=tf.float32)

    anchor_xy = y_true[..., 0:2]
    true_wh = y_true[..., 2:4]

    # scale thhe boxes
    anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
    anchors_x = anchors[..., 0] / width
    anchors_y = anchors[..., 1] / height
    anchors = tf.stack([anchors_x, anchors_y], axis=-1)

    # build a matrix of anchor boxes
    anchors = tf.transpose(anchors, perm=[1, 0])
    anchor_xy = tf.tile(tf.expand_dims(anchor_xy, axis=-1),
                        [1, 1, tf.shape(anchors)[-1]])
    anchors = tf.tile(tf.expand_dims(anchors, axis=0),
                      [tf.shape(anchor_xy)[0], 1, 1])

    # stack the xy so, each anchor is asscoaited once with each center from
    # the ground truth input
    anchors = K.concatenate([anchor_xy, anchors], axis=1)
    anchors = tf.transpose(anchors, perm=[2, 0, 1])

    # copy the gt n times so that each anchor from above can be compared to
    # input ground truth
    truth_comp = tf.tile(tf.expand_dims(y_true[..., 0:4], axis=-1),
                         [1, 1, tf.shape(anchors)[0]])
    truth_comp = tf.transpose(truth_comp, perm=[2, 0, 1])

    # compute intersection over union of the boxes, and take the argmax of
    # comuted iou for each box. thus each box is associated with the largest
    # interection over union
    iou_raw = compute_iou(truth_comp, anchors)

    gt_mask = tf.cast(iou_raw > 0.213, dtype=iou_raw.dtype)

    num_k = tf.reduce_max(
        tf.reduce_sum(tf.transpose(gt_mask, perm=[1, 0]), axis=1))
    if num_k <= 0:
      num_k = 1.0

    values, indexes = tf.math.top_k(tf.transpose(iou_raw, perm=[1, 0]),
                                    k=tf.cast(num_k, dtype=tf.int32),
                                    sorted=True)
    ind_mask = tf.cast(values > 0.213, dtype=indexes.dtype)
    iou_index = tf.concat([
        K.expand_dims(indexes[..., 0], axis=-1),
        ((indexes[..., 1:] + 1) * ind_mask[..., 1:]) - 1
    ],
                          axis=-1)

    stack = tf.zeros(
        [tf.shape(iou_index)[0],
         tf.cast(1, dtype=iou_index.dtype)],
        dtype=iou_index.dtype) - 1
    while num_k < 5:
      iou_index = tf.concat([iou_index, stack], axis=-1)
      num_k += 1
    iou_index = iou_index[..., :5]

    values = tf.concat([
        K.expand_dims(values[..., 0], axis=-1),
        ((values[..., 1:]) * tf.cast(ind_mask[..., 1:], dtype=tf.float32))
    ],
                       axis=-1)
  return tf.cast(iou_index, dtype=tf.float32)
