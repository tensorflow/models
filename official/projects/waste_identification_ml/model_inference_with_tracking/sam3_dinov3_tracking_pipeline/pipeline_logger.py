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

"""Logging configuration and handlers for the SAM3/DINOv3 tracking pipeline."""

import logging
import pathlib
import sys

_LOGGER_NAME = "waste_identification_pipeline"
_LOG_FORMAT = "[%(asctime)s] %(levelname)s %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger() -> logging.Logger:
  """Returns the shared, configured pipeline logger.

  The logger writes to stdout, bypasses third-party root loggers
  (`propagate=False`), and is idempotent: repeated calls return the same
  instance without adding duplicate handlers.

  Returns:
    The shared pipeline logger instance.
  """
  logger = logging.getLogger(_LOGGER_NAME)
  if logger.handlers:
    return logger
  logger.setLevel(logging.INFO)
  stream_handler = logging.StreamHandler(stream=sys.stdout)
  stream_handler.setFormatter(
      logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
  )
  logger.addHandler(stream_handler)
  logger.propagate = False
  return logger


def attach_file_handler(log_file_path: pathlib.Path) -> logging.FileHandler:
  """Attaches a file handler to the pipeline logger.

  The new handler inherits the logger's current level and reuses the stream
  handler's formatter so file logs match console logs. If no stream handler
  is present, the default pipeline formatter is used instead.

  Args:
    log_file_path: Path where the log file will be written.

  Returns:
    The attached FileHandler so the caller can detach it later.
  """
  logger = get_logger()
  file_handler = logging.FileHandler(
      str(log_file_path), mode="w", encoding="utf-8"
  )
  file_handler.setLevel(logger.level)
  file_handler.setFormatter(_get_stream_formatter(logger))
  logger.addHandler(file_handler)
  return file_handler


def detach_file_handler(file_handler: logging.FileHandler) -> None:
  """Removes a file handler from the pipeline logger and closes it.

  Args:
    file_handler: The handler previously returned by `attach_file_handler`.
  """
  logger = get_logger()
  logger.removeHandler(file_handler)
  file_handler.close()


def _get_stream_formatter(logger: logging.Logger) -> logging.Formatter:
  """Returns the formatter used by the first StreamHandler on the logger.

  Falls back to the default pipeline formatter if no StreamHandler is
  attached or the attached one has no formatter.

  Args:
    logger: Logger to inspect.

  Returns:
    A logging.Formatter suitable for consistent log output.
  """
  for handler in logger.handlers:
    if (
        isinstance(handler, logging.StreamHandler)
        and handler.formatter is not None
    ):
      return handler.formatter
  return logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT)
