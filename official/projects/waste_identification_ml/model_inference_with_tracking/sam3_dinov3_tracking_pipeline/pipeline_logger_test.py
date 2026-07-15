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

"""Unit tests for pipeline_logger.py."""

import logging
import pathlib

from absl.testing import absltest
from official.projects.waste_identification_ml.model_inference_with_tracking.sam3_dinov3_tracking_pipeline import pipeline_logger


class GetLoggerTest(absltest.TestCase):
  """Tests for get_logger."""

  def setUp(self):
    """Detaches all handlers before each test so the logger starts clean."""
    super().setUp()
    logger = logging.getLogger(pipeline_logger._LOGGER_NAME)
    # Iterate over a shallow copy (`list(...)`) so removing handlers does not
    # mutate the list during iteration.
    for handler in list(logger.handlers):
      logger.removeHandler(handler)
    self.addCleanup(logger.handlers.clear)

  def test_returns_configured_logger_with_stream_handler(self):
    """Verifies a first call attaches exactly one StreamHandler."""
    logger = pipeline_logger.get_logger()
    stream_handlers = [
        handler
        for handler in logger.handlers
        if isinstance(handler, logging.StreamHandler)
    ]
    self.assertLen(stream_handlers, 1)
    self.assertEqual(logger.level, logging.INFO)
    self.assertFalse(logger.propagate)

  def test_repeated_calls_do_not_duplicate_handlers(self):
    """Verifies the logger is idempotent across many get_logger calls."""
    first = pipeline_logger.get_logger()
    handler_count = len(first.handlers)
    for _ in range(5):
      pipeline_logger.get_logger()
    self.assertLen(first.handlers, handler_count)


class AttachAndDetachFileHandlerTest(absltest.TestCase):
  """Tests for attach_file_handler and detach_file_handler."""

  def setUp(self):
    """Resets the pipeline logger before each test."""
    super().setUp()
    logger = logging.getLogger(pipeline_logger._LOGGER_NAME)
    # Iterate over a shallow copy (`list(...)`) so removing handlers does not
    # mutate the list during iteration.
    for handler in list(logger.handlers):
      logger.removeHandler(handler)
    self.addCleanup(logger.handlers.clear)

  def test_attach_adds_file_handler_with_matching_formatter(self):
    """Verifies the attached FileHandler reuses the stream handler formatter."""
    pipeline_logger.get_logger()  # Ensure a stream handler is present.
    log_path = pathlib.Path(self.create_tempdir().full_path) / "run.log"

    file_handler = pipeline_logger.attach_file_handler(log_path)
    try:
      logger = pipeline_logger.get_logger()
      self.assertIn(file_handler, logger.handlers)
      stream_handler = next(
          handler
          for handler in logger.handlers
          if isinstance(handler, logging.StreamHandler)
          and not isinstance(handler, logging.FileHandler)
      )
      self.assertIs(file_handler.formatter, stream_handler.formatter)
      self.assertEqual(file_handler.level, logger.level)
    finally:
      pipeline_logger.detach_file_handler(file_handler)

  def test_detach_removes_and_closes_handler(self):
    """Verifies detach_file_handler removes the handler and closes it."""
    pipeline_logger.get_logger()
    log_path = pathlib.Path(self.create_tempdir().full_path) / "run.log"
    file_handler = pipeline_logger.attach_file_handler(log_path)

    stream = file_handler.stream
    self.assertIsNotNone(stream)
    pipeline_logger.detach_file_handler(file_handler)

    logger = pipeline_logger.get_logger()
    self.assertNotIn(file_handler, logger.handlers)
    self.assertTrue(stream.closed)
    self.assertIsNone(file_handler.stream)


if __name__ == "__main__":
  absltest.main()
