# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
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

"""TFDS factory functions."""
from official.vision.beta.dataloaders import decoder as base_decoder
from official.vision.beta.dataloaders import tfds_detection_decoders
from official.vision.beta.dataloaders import tfds_segmentation_decoders
from official.vision.beta.dataloaders import tfds_classification_decoders


def get_classification_decoder(tfds_name: str) -> base_decoder.Decoder:
  """Gets classification decoder.

  Args:
    tfds_name: `str`, name of the tfds classification decoder.
  Returns:
    `base_decoder.Decoder` instance.
  Raises:
    ValueError if the tfds_name doesn't exist in the available decoders.
  """
  if tfds_name in tfds_classification_decoders.TFDS_ID_TO_DECODER_MAP:
    decoder = tfds_classification_decoders.TFDS_ID_TO_DECODER_MAP[tfds_name]()
  else:
    raise ValueError(
        f'TFDS Classification {tfds_name} is not supported')
  return decoder


def get_detection_decoder(tfds_name: str) -> base_decoder.Decoder:
  """Gets detection decoder.

  Args:
    tfds_name: `str`, name of the tfds detection decoder.
  Returns:
    `base_decoder.Decoder` instance.
  Raises:
    ValueError if the tfds_name doesn't exist in the available decoders.
  """
  if tfds_name in tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP:
    decoder = tfds_detection_decoders.TFDS_ID_TO_DECODER_MAP[tfds_name]()
  else:
    raise ValueError(f'TFDS Detection {tfds_name} is not supported')
  return decoder


def get_segmentation_decoder(tfds_name: str) -> base_decoder.Decoder:
  """Gets segmentation decoder.

  Args:
    tfds_name: `str`, name of the tfds segmentation decoder.
  Returns:
    `base_decoder.Decoder` instance.
  Raises:
    ValueError if the tfds_name doesn't exist in the available decoders.
  """
  if tfds_name in tfds_segmentation_decoders.TFDS_ID_TO_DECODER_MAP:
    decoder = tfds_segmentation_decoders.TFDS_ID_TO_DECODER_MAP[tfds_name]()
  else:
    raise ValueError(f'TFDS Segmentation {tfds_name} is not supported')
  return decoder
