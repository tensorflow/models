# Copyright 2026 The TensorFlow Authors. All Rights Reserved.
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

"""Tests for datasets and data loaders in DINOv3 image classifier."""

import os
import tempfile
import unittest
from absl.testing import parameterized
from PIL import Image
import torch
from torch.utils import data as torch_data
from torchvision import datasets as tv_datasets
from torchvision.transforms import v2
from official.projects.waste_identification_ml.fine_tuning.Dinov3_image_classifier import datasets


class DatasetsTest(parameterized.TestCase):
  """Test suite for DINOv3 dataset and data loader utilities."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()
    self.image_size = 64
    self.image_mean = (0.485, 0.456, 0.406)
    self.image_std = (0.229, 0.224, 0.225)

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def _create_dummy_image_folder(self, base_dir: str) -> str:
    """Helper to create dummy ImageFolder with synthetic images."""
    for class_name in ['cardboard', 'plastic']:
      class_dir = os.path.join(base_dir, class_name)
      os.makedirs(class_dir, exist_ok=True)
      for i in range(2):
        img_path = os.path.join(class_dir, f'image_{i}.jpg')
        # Create an RGB image of size (100, 80)
        img = Image.new('RGB', (100, 80), color=(100 + i * 20, 150, 200))
        img.save(img_path)
    return base_dir

  @parameterized.named_parameters(
      ('train', datasets._get_train_transform, (100, 100), (128, 128, 128)),
      ('valid', datasets._get_valid_transform, (120, 90), (50, 100, 150)),
  )
  def test_transform_pipeline(self, transform_fn, image_dims, image_color):
    transform = transform_fn(self.image_size, self.image_mean, self.image_std)
    self.assertIsInstance(transform, v2.Compose)

    # Test transform execution on a raw PIL image.
    dummy_image = Image.new('RGB', image_dims, color=image_color)
    tensor_out = transform(dummy_image)

    self.assertIsInstance(tensor_out, torch.Tensor)
    self.assertEqual(tensor_out.shape, (3, self.image_size, self.image_size))
    self.assertEqual(tensor_out.dtype, torch.float32)

  def test_get_datasets_loads_train_and_valid_folders(self):
    train_dir = self._create_dummy_image_folder(
        os.path.join(self.temp_dir.name, 'train')
    )
    valid_dir = self._create_dummy_image_folder(
        os.path.join(self.temp_dir.name, 'valid')
    )

    dataset_train, dataset_valid, class_names = datasets.get_datasets(
        train_dir=train_dir,
        valid_dir=valid_dir,
        image_size=self.image_size,
        image_mean=self.image_mean,
        image_std=self.image_std,
    )

    self.assertIsInstance(dataset_train, tv_datasets.ImageFolder)
    self.assertIsInstance(dataset_valid, tv_datasets.ImageFolder)
    self.assertLen(dataset_train, 4)  # 2 classes * 2 images
    self.assertLen(dataset_valid, 4)
    self.assertEqual(class_names, ['cardboard', 'plastic'])

    # Verify item retrieval outputs transformed tensors and integer class
    # indices.
    img, label = dataset_train[0]
    self.assertIsInstance(img, torch.Tensor)
    self.assertEqual(img.shape, (3, self.image_size, self.image_size))
    self.assertIsInstance(label, int)

  @parameterized.named_parameters(
      ('zero_workers', 0, False, None),
      ('with_workers', 1, True, 4),
  )
  def test_get_data_loaders_configuration(
      self, num_workers, expected_persistent_workers, expected_prefetch_factor
  ):
    train_dir = self._create_dummy_image_folder(
        os.path.join(self.temp_dir.name, 'train')
    )
    valid_dir = self._create_dummy_image_folder(
        os.path.join(self.temp_dir.name, 'valid')
    )
    dataset_train, dataset_valid, _ = datasets.get_datasets(
        train_dir=train_dir,
        valid_dir=valid_dir,
        image_size=self.image_size,
        image_mean=self.image_mean,
        image_std=self.image_std,
    )

    batch_size = 2
    train_loader, valid_loader = datasets.get_data_loaders(
        dataset_train=dataset_train,
        dataset_valid=dataset_valid,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    self.assertIsInstance(train_loader, torch_data.DataLoader)
    self.assertIsInstance(valid_loader, torch_data.DataLoader)
    self.assertEqual(train_loader.batch_size, batch_size)
    self.assertEqual(valid_loader.batch_size, batch_size)
    self.assertEqual(
        train_loader.persistent_workers, expected_persistent_workers
    )
    self.assertEqual(train_loader.prefetch_factor, expected_prefetch_factor)
    self.assertEqual(
        valid_loader.persistent_workers, expected_persistent_workers
    )
    self.assertEqual(valid_loader.prefetch_factor, expected_prefetch_factor)

    if num_workers == 0:
      # Verify iteration over train_loader produces batches of expected shape.
      batch_x, batch_y = next(iter(train_loader))
      self.assertEqual(
          batch_x.shape, (batch_size, 3, self.image_size, self.image_size)
      )
      self.assertEqual(batch_y.shape, (batch_size,))


if __name__ == '__main__':
  unittest.main()
