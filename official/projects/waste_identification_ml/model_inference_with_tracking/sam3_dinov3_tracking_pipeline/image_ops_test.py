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

"""Unit tests for image_ops.py."""

import pathlib
from unittest import mock

from absl.testing import absltest
import numpy
from PIL import Image
from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import image_ops


class LoadAndResizeImageTest(absltest.TestCase):
  """Tests for load_and_resize_image."""

  def setUp(self):
    """Patches cv2 read/resize/cvtColor for deterministic behavior."""
    super().setUp()
    self.mock_imread = self.enter_context(
        mock.patch.object(image_ops.cv2, "imread", autospec=True)
    )
    self.mock_resize = self.enter_context(
        mock.patch.object(image_ops.cv2, "resize", autospec=True)
    )
    self.mock_cvt_color = self.enter_context(
        mock.patch.object(image_ops.cv2, "cvtColor", autospec=True)
    )
    # cvtColor always returns a small deterministic RGB array.
    self.mock_cvt_color.return_value = numpy.zeros((4, 4, 3), dtype=numpy.uint8)

  def test_raises_image_read_error_when_cv2_returns_none(self):
    """Verifies ImageReadError is raised when imread cannot decode the file."""
    self.mock_imread.return_value = None
    with self.assertRaisesRegex(
        image_ops.ImageReadError, "OpenCV could not read the image"
    ):
      image_ops.load_and_resize_image(
          image_path=pathlib.Path("/nowhere/x.png"), max_short_side=800
      )

  def test_skips_resize_when_image_is_small_enough(self):
    """Verifies cv2.resize is not called when short side is within limit."""
    self.mock_imread.return_value = numpy.zeros(
        (100, 200, 3), dtype=numpy.uint8
    )
    result = image_ops.load_and_resize_image(
        image_path=pathlib.Path("/x/y.png"), max_short_side=800
    )
    self.mock_resize.assert_not_called()
    self.assertIsInstance(result, Image.Image)

  def test_resizes_when_short_side_exceeds_max(self):
    """Verifies cv2.resize is called with the scaled dimensions."""
    # Input is 1000 x 2000, short side is 1000, max is 400 -> scale = 0.4.
    self.mock_imread.return_value = numpy.zeros(
        (1000, 2000, 3), dtype=numpy.uint8
    )
    self.mock_resize.return_value = numpy.zeros(
        (400, 800, 3), dtype=numpy.uint8
    )
    image_ops.load_and_resize_image(
        image_path=pathlib.Path("/x/y.png"), max_short_side=400
    )
    self.mock_resize.assert_called_once()
    passed_size = self.mock_resize.call_args.args[1]
    self.assertEqual(passed_size, (800, 400))


class ComputeResizedDimensionsTest(absltest.TestCase):
  """Tests for the private _compute_resized_dimensions helper."""

  def test_returns_original_dimensions_when_within_limit(self):
    """Verifies dimensions are returned unchanged when already small enough."""
    self.assertEqual(
        image_ops._compute_resized_dimensions(
            width=200, height=100, max_short_side=800
        ),
        (200, 100),
    )

  def test_scales_down_when_short_side_exceeds_limit(self):
    """Verifies scale factor is applied to both dimensions."""
    # 2000 x 1000, short = 1000, limit = 400 -> scale = 0.4 -> (800, 400).
    self.assertEqual(
        image_ops._compute_resized_dimensions(
            width=2000, height=1000, max_short_side=400
        ),
        (800, 400),
    )

  def test_preserves_aspect_ratio_when_scaling(self):
    """Verifies the ratio width/height is preserved after scaling."""
    new_width, new_height = image_ops._compute_resized_dimensions(
        width=1600, height=900, max_short_side=450
    )
    # Short side becomes 450; long side scales by the same factor.
    self.assertEqual((new_width, new_height), (800, 450))


class EnsureDirectoriesExistTest(absltest.TestCase):
  """Tests for ensure_directories_exist."""

  def test_creates_each_directory(self):
    """Verifies each real directory path is created on the filesystem."""
    temp_root = pathlib.Path(self.create_tempdir().full_path)
    dir_a = temp_root / "subdir_a"
    dir_b = temp_root / "nested" / "subdir_b"

    image_ops.ensure_directories_exist([dir_a, dir_b])

    self.assertTrue(dir_a.exists())
    self.assertTrue(dir_b.exists())
    self.assertTrue(dir_a.is_dir())
    self.assertTrue(dir_b.is_dir())

  def test_raises_value_error_on_empty_or_current_dir_path(self):
    """Verifies empty paths and '.' are caught as config errors."""
    with self.assertRaisesRegex(
        ValueError, "empty path or the current working directory"
    ):
      image_ops.ensure_directories_exist([pathlib.Path("")])
    with self.assertRaisesRegex(
        ValueError, "empty path or the current working directory"
    ):
      image_ops.ensure_directories_exist([pathlib.Path(".")])


class ReleaseCachesTest(absltest.TestCase):
  """Tests for release_caches."""

  def test_runs_gc_and_skips_cuda_when_unavailable(self):
    """Verifies gc.collect runs and cuda.empty_cache is skipped without GPU."""
    mock_gc_collect = self.enter_context(
        mock.patch.object(image_ops.gc, "collect", autospec=True)
    )
    self.enter_context(
        mock.patch.object(
            image_ops.torch.cuda,
            "is_available",
            autospec=True,
            return_value=False,
        )
    )
    mock_empty_cache = self.enter_context(
        mock.patch.object(image_ops.torch.cuda, "empty_cache", autospec=True)
    )

    image_ops.release_caches()

    mock_gc_collect.assert_called_once()
    mock_empty_cache.assert_not_called()

  def test_calls_cuda_empty_cache_when_available(self):
    """Verifies cuda.empty_cache is called when CUDA is available."""
    self.enter_context(
        mock.patch.object(image_ops.gc, "collect", autospec=True)
    )
    self.enter_context(
        mock.patch.object(
            image_ops.torch.cuda,
            "is_available",
            autospec=True,
            return_value=True,
        )
    )
    mock_empty_cache = self.enter_context(
        mock.patch.object(image_ops.torch.cuda, "empty_cache", autospec=True)
    )

    image_ops.release_caches()

    mock_empty_cache.assert_called_once()


if __name__ == "__main__":
  absltest.main()
