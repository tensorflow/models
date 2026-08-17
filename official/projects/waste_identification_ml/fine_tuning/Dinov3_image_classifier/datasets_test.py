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
import pathlib
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


class ComputeBalancedClassWeightsTest(parameterized.TestCase):
  """Test suite for the compute_balanced_class_weights helper."""

  def setUp(self):
    super().setUp()
    self.temp_dir = tempfile.TemporaryDirectory()

  def tearDown(self):
    self.temp_dir.cleanup()
    super().tearDown()

  def _create_class_directory(
      self, base_dir: pathlib.Path, class_name: str, number_of_files: int
  ) -> None:
    """Creates a class subdirectory populated with dummy files.

    Args:
      base_dir: The parent directory that will contain the class folder.
      class_name: Name of the class subdirectory to create.
      number_of_files: How many placeholder files to write into the class
        folder.
    """
    class_directory = base_dir / class_name
    class_directory.mkdir(parents=True, exist_ok=True)
    for file_index in range(number_of_files):
      (class_directory / f'sample_{file_index}.jpg').write_bytes(b'')

  def test_returns_balanced_weights_for_equal_class_counts(self):
    """Verifies equal counts yield equal weights of value 1.0."""
    train_directory = pathlib.Path(self.temp_dir.name)
    self._create_class_directory(train_directory, 'cardboard', 4)
    self._create_class_directory(train_directory, 'plastic', 4)

    weights = datasets.compute_balanced_class_weights(train_directory)

    self.assertEqual(weights.dtype, torch.float32)
    self.assertEqual(weights.shape, (2,))
    torch.testing.assert_close(
        weights, torch.tensor([1.0, 1.0], dtype=torch.float32)
    )

  def test_returns_higher_weight_for_minority_class(self):
    """Verifies the rarer class receives a proportionally larger weight."""
    train_directory = pathlib.Path(self.temp_dir.name)
    # 8 cardboard vs 2 plastic. Total = 10, number_of_classes = 2.
    # cardboard weight = 10 / (2 * 8) = 0.625
    # plastic   weight = 10 / (2 * 2) = 2.5
    self._create_class_directory(train_directory, 'cardboard', 8)
    self._create_class_directory(train_directory, 'plastic', 2)

    weights = datasets.compute_balanced_class_weights(train_directory)

    self.assertEqual(weights.shape, (2,))
    torch.testing.assert_close(
        weights, torch.tensor([0.625, 2.5], dtype=torch.float32)
    )

  def test_class_ordering_is_alphabetical(self):
    """Verifies weights are ordered alphabetically by class name."""
    train_directory = pathlib.Path(self.temp_dir.name)
    # Create in non-alphabetical order to prove sorting takes effect.
    self._create_class_directory(train_directory, 'plastic', 1)
    self._create_class_directory(train_directory, 'cardboard', 4)

    weights = datasets.compute_balanced_class_weights(train_directory)

    # Alphabetical: cardboard first, plastic second.
    # cardboard weight = 5 / (2 * 4) = 0.625
    # plastic   weight = 5 / (2 * 1) = 2.5
    torch.testing.assert_close(
        weights, torch.tensor([0.625, 2.5], dtype=torch.float32)
    )

  def test_raises_when_no_class_subdirectories(self):
    """Verifies an empty train directory raises ValueError."""
    train_directory = pathlib.Path(self.temp_dir.name)

    with self.assertRaisesRegex(ValueError, 'No class subdirectories'):
      datasets.compute_balanced_class_weights(train_directory)

  def test_raises_when_a_class_directory_is_empty(self):
    """Verifies an empty class subdirectory raises ValueError."""
    train_directory = pathlib.Path(self.temp_dir.name)
    self._create_class_directory(train_directory, 'cardboard', 4)
    # Create empty plastic directory (no files).
    (train_directory / 'plastic').mkdir()

    with self.assertRaisesRegex(ValueError, 'empty'):
      datasets.compute_balanced_class_weights(train_directory)

  def test_ignores_files_at_top_level(self):
    """Verifies stray non-directory entries at the top level are skipped."""
    train_directory = pathlib.Path(self.temp_dir.name)
    self._create_class_directory(train_directory, 'cardboard', 4)
    self._create_class_directory(train_directory, 'plastic', 4)
    # A stray file next to the class directories.
    (train_directory / 'README.txt').write_text('not a class')

    weights = datasets.compute_balanced_class_weights(train_directory)

    # Should still see only the two class directories.
    self.assertEqual(weights.shape, (2,))

  def test_ignores_subdirectories_inside_class_folder(self):
    """Verifies nested directories inside a class folder are not counted."""
    train_directory = pathlib.Path(self.temp_dir.name)
    self._create_class_directory(train_directory, 'cardboard', 4)
    self._create_class_directory(train_directory, 'plastic', 4)
    # A nested directory that should NOT be counted as a file.
    (train_directory / 'cardboard' / 'thumbnails').mkdir()

    weights = datasets.compute_balanced_class_weights(train_directory)

    # Both classes still have 4 files each, so weights remain balanced.
    torch.testing.assert_close(
        weights, torch.tensor([1.0, 1.0], dtype=torch.float32)
    )


if __name__ == '__main__':
  unittest.main()
