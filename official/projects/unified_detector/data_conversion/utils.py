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

"""Utilities to convert data to TFExamples and store in TFRecords."""

from typing import Any, Dict, List, Tuple, Union


import cv2
import numpy as np
import tensorflow as tf, tf_keras


def encode_image(
    image_tensor: np.ndarray,
    encoding_type: str = 'png') -> Union[np.ndarray, tf.Tensor]:
  """Encode image tensor into byte string."""
  if encoding_type == 'jpg':
    image_encoded = tf.image.encode_jpeg(tf.constant(image_tensor))
  elif encoding_type == 'png':
    image_encoded = tf.image.encode_png(tf.constant(image_tensor))
  else:
    raise ValueError('Invalid encoding type.')
  if tf.executing_eagerly():
    image_encoded = image_encoded.numpy()
  else:
    image_encoded = image_encoded.eval()
  return image_encoded


def int64_feature(value: Union[int, List[int]]) -> tf.train.Feature:
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value: Union[float, List[float]]) -> tf.train.Feature:
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value: Union[Union[bytes, str], List[Union[bytes, str]]]
                  ) -> tf.train.Feature:
  if not isinstance(value, list):
    value = [value]
  for i in range(len(value)):
    if not isinstance(value[i], bytes):
      value[i] = value[i].encode('utf-8')
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def annotation_to_entities(annotation: Dict[str, Any]) -> List[Dict[str, Any]]:
  """Flatten the annotation dict to a list of 'entities'."""
  entities = []
  for paragraph in annotation['paragraphs']:
    paragraph_id = len(entities)
    paragraph['type'] = 3  # 3 for paragraph
    paragraph['parent_id'] = -1
    entities.append(paragraph)

    for line in paragraph['lines']:
      line_id = len(entities)
      line['type'] = 2  # 2 for line
      line['parent_id'] = paragraph_id
      entities.append(line)

      for word in line['words']:
        word['type'] = 1  # 1 for word
        word['parent_id'] = line_id
        entities.append(word)

  return entities


def draw_entity_mask(
    entities: List[Dict[str, Any]],
    image_shape: Tuple[int, int, int]) -> np.ndarray:
  """Draw entity id mask.

  Args:
    entities: A list of entity objects. Should be output from
      `annotation_to_entities`.
    image_shape: The shape of the input image.
  Returns:
    A (H, W, 3) entity id mask of the same height/width as the image. Each pixel
    (i, j, :) encodes the entity id of one pixel. Only word entities are
    rendered. 0 for non-text pixels; word entity ids start from 1.
  """
  instance_mask = np.zeros(image_shape, dtype=np.uint8)
  for i, entity in enumerate(entities):
    # only draw word masks
    if entity['type'] != 1:
      continue
    vertices = np.array(entity['vertices'])
    # the pixel value is actually 1 + position in entities
    entity_id = i + 1
    if entity_id >= 65536:
      # As entity_id is encoded in the last two channels, it should be less than
      # 256**2=65536.
      raise ValueError(
          (f'Entity ID overflow: {entity_id}. Currently only entity_id<65536 '
           'are supported.'))

    # use the last two channels to encode the entity id.
    color = [0, entity_id // 256, entity_id % 256]
    instance_mask = cv2.fillPoly(instance_mask,
                                 [np.round(vertices).astype('int32')], color)
  return instance_mask


def convert_to_tfe(img_file_name: str,
                   annotation: Dict[str, Any]) -> tf.train.Example:
  """Convert the annotation dict into a TFExample."""

  img = cv2.imread(img_file_name)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  h, w, c = img.shape
  encoded_img = encode_image(img)

  entities = annotation_to_entities(annotation)
  masks = draw_entity_mask(entities, img.shape)
  encoded_mask = encode_image(masks)

  # encode attributes
  parent = []
  classes = []
  content_type = []
  text = []
  vertices = []

  for entity in entities:
    parent.append(entity['parent_id'])
    classes.append(entity['type'])
    # 0 for annotated; 8 for not annotated
    content_type.append((0 if entity['legible'] else 8))
    text.append(entity.get('text', ''))
    v = np.array(entity['vertices'])
    vertices.append(','.join(str(float(n)) for n in v.reshape(-1)))

  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              # input images
              'image/encoded': bytes_feature(encoded_img),
              # image format
              'image/format': bytes_feature('png'),
              # image width
              'image/width': int64_feature([w]),
              # image height
              'image/height': int64_feature([h]),
              # image channels
              'image/channels': int64_feature([c]),
              # image key
              'image/source_id': bytes_feature(annotation['image_id']),
              # HxWx3 tensors: channel 2-3 encodes the id of the word entity.
              'image/additional_channels/encoded': bytes_feature(encoded_mask),
              # format of the additional channels
              'image/additional_channels/format': bytes_feature('png'),
              'image/object/parent': int64_feature(parent),
              # word / line / paragraph / symbol / ...
              'image/object/classes': int64_feature(classes),
              # text / handwritten / not-annotated / ...
              'image/object/content_type': int64_feature(content_type),
              # string text transcription
              'image/object/text': bytes_feature(text),
              # comma separated coordinates, (x,y) * n
              'image/object/vertices': bytes_feature(vertices),
          })).SerializeToString()

  return example
