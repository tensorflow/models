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
"""Pipeline to run the prediction on the images folder."""

import os

from absl import app
from absl import flags
from big_query_ops import BigQueryManager
from constants import BATCH_SIZE
from constants import BOTTLE_EXTRACTION_CONFIDENCE_THRESHOLD
from constants import BOTTLE_EXTRACTION_CONTAINMENT_THRESHOLD
from constants import BOTTLE_EXTRACTION_CROP_SIZE
from constants import BOTTLE_EXTRACTION_MAX_SHORT_SIDE
from constants import BOTTLE_EXTRACTION_SCORE_THRESHOLD
from constants import BYTETRACK_MINIMUM_CONSECUTIVE_FRAMES
from constants import BYTETRACK_MINIMUM_IOU_THRESHOLD
from constants import CLASS_NAMES
from constants import CLASSIFICATION_THRESHOLD
from constants import CLASSIFIER_CHECKPOINT_PATH
from constants import DETECTION_PROMPT
from constants import DINOV3_MODEL_NAME
from constants import DINOV3_REPO_DIR
from constants import SAM3_CHECKPOINT_PATH
from object_extraction import ObjectExtractor
import pandas as pd
from pet_grade_classifier import DinoClassifier
from PIL import Image
import tqdm
from trackers import ByteTrackTracker
import utils

INPUT_DIRECTORY = flags.DEFINE_string(
    "input_directory", None, "The path to the directory containing images."
)
OUTPUT_DIRECTORY = flags.DEFINE_string(
    "output_directory", None, "The path to the directory to save the results."
)

OVERWRITE = flags.DEFINE_boolean(
    "overwrite",
    False,
    "If True, delete the preexisting BigQuery table before creating a new one.",
)
PROJECT_ID = flags.DEFINE_string(
    "project_id", None, "Project ID mentioned in Google Cloud Project"
)
BQ_DATASET_ID = flags.DEFINE_string(
    "bq_dataset_id", "Circularnet_dataset", "Big query dataset ID"
)
BQ_TABLE_ID = flags.DEFINE_string(
    "bq_table_id", "Circularnet_table", "BigQuery Table ID for features data"
)


object_extractor = ObjectExtractor(
    checkpoint_path=SAM3_CHECKPOINT_PATH,
    confidence_threshold=BOTTLE_EXTRACTION_CONFIDENCE_THRESHOLD,
    score_threshold=BOTTLE_EXTRACTION_SCORE_THRESHOLD,
    containment_threshold=BOTTLE_EXTRACTION_CONTAINMENT_THRESHOLD,
    max_short_side=BOTTLE_EXTRACTION_MAX_SHORT_SIDE,
    crop_size=BOTTLE_EXTRACTION_CROP_SIZE,
)


classifier = DinoClassifier(
    classifier_checkpoint_path=CLASSIFIER_CHECKPOINT_PATH,
    dinov3_repo_dir=DINOV3_REPO_DIR,
    model_name=DINOV3_MODEL_NAME,
    class_names=CLASS_NAMES,
)


tracker = ByteTrackTracker(
    minimum_iou_threshold=BYTETRACK_MINIMUM_IOU_THRESHOLD,
    minimum_consecutive_frames=BYTETRACK_MINIMUM_CONSECUTIVE_FRAMES,
)


def get_batch_of_crops(image_paths, tracker_instance, batch_size=10):
  """Yields per-batch crop dicts; tracker_instance persists across batches.

  Args:
    image_paths: List of file paths to the images.
    tracker_instance: The tracker instance that persists across batches.
    batch_size: Number of images to process in each batch.

  Yields:
    A dict mapping tracker ID to a list of dicts with frame name and crop.
    One dict per batch of batch_size frames. The dict only contains
    crops collected in that batch — not accumulated across all batches.
  """
  for batch_start in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[batch_start : batch_start + batch_size]

    batch_records = {}
    for image_path in tqdm.tqdm(
        batch_paths,
        desc=f"Batch {batch_start//batch_size + 1} frames",
        leave=False,
    ):
      rgb_image = Image.open(image_path).convert("RGB")
      resized_image, state, detections = object_extractor.extract(
          rgb_image, prompt=DETECTION_PROMPT
      )
      detections = tracker_instance.update(detections)
      object_extractor.get_cropped_objects_for_each_tracking_id(
          image=resized_image,
          state=state,
          detections=detections,
          source_frame_name=os.path.basename(image_path),
          crop_size=BOTTLE_EXTRACTION_CROP_SIZE,
          track_crop_records=batch_records,
      )

    yield batch_records


def main(_) -> None:
  if (
      not INPUT_DIRECTORY.value
      or not OUTPUT_DIRECTORY.value
      or not INPUT_DIRECTORY.value.startswith("gs://")
      or not OUTPUT_DIRECTORY.value.startswith("gs://")
  ):
    raise ValueError("Bucket path must be non-empty starting with 'gs://'")

  input_directory, prediction_folder, logger = (
      utils.setup_logger_and_directories(input_dir=INPUT_DIRECTORY.value)
  )

  checkpoint_path = os.path.join(prediction_folder, "prediction.csv")

  storage_manager = BigQueryManager(
      project_id=PROJECT_ID.value,
      dataset_id=BQ_DATASET_ID.value,
      table_id=BQ_TABLE_ID.value,
  )

  filepaths = utils.files_paths(os.path.basename(input_directory))
  num_batches = (len(filepaths) + BATCH_SIZE - 1) // BATCH_SIZE
  logger.info(
      f"Found {len(filepaths)} image files. Starting inference over"
      f" {num_batches} batches."
  )
  for batch_records in tqdm.tqdm(
      get_batch_of_crops(
          image_paths=filepaths, tracker_instance=tracker, batch_size=BATCH_SIZE
      ),
      total=num_batches,
      desc="Batches",
  ):
    batch_predictions = []
    for tracker_id, crop_records in tqdm.tqdm(
        batch_records.items(), desc="Classifying crops", leave=False
    ):
      crop_pil_images = [crop_record["crop"] for crop_record in crop_records]
      crop_predictions = classifier.predict(pil_images=crop_pil_images)
      for crop_record, prediction in zip(crop_records, crop_predictions):
        batch_predictions.append({
            "tracker_id": tracker_id,
            "frame_name": crop_record["frame_name"],
            "crop": crop_record["crop"],
            "predicted_class": prediction["predicted_class"],
            "predicted_probability": prediction["predicted_probability"],
        })
    batch_predictions = utils.get_class_with_majority_vote(
        batch_predictions, min_probability=CLASSIFICATION_THRESHOLD
    )
    utils.save_output_image_grids(
        all_predictions=batch_predictions, output_dr=prediction_folder
    )

    # Append this image's rows to checkpoint immediately
    if batch_predictions:
      rows_df = (
          pd.DataFrame(batch_predictions)
          .drop(columns=["crop", "predicted_class", "predicted_probability"])
          .drop_duplicates(subset=["tracker_id"])
      )
      rows_df.to_csv(
          checkpoint_path,
          mode="a",
          header=not os.path.exists(checkpoint_path)
          or os.path.getsize(checkpoint_path) == 0,
          index=False,
      )

  # Upload Results to BigQuery
  results_df = (
      pd.read_csv(checkpoint_path).dropna()
      if os.path.exists(checkpoint_path)
      else pd.DataFrame()
  )
  if isinstance(results_df, pd.DataFrame) and not results_df.empty:
    try:
      logger.info(
          f"Starting BigQuery ingestion for {len(results_df)} records..."
      )
      storage_manager.create_table(overwrite=OVERWRITE.value)
      storage_manager.ingest_data(results_df)
      storage_manager.upload_image_results_to_storage_bucket(
          input_directory=input_directory,
          prediction_folder=prediction_folder,
          output_directory=OUTPUT_DIRECTORY.value,
      )
      logger.info("Pipeline execution successfully completed.")
    except (KeyError, IndexError, TypeError, ValueError) as e:
      logger.info(f"Issue in creation of table, due to error : {e}")
  else:
    logger.info("No data to ingest.")
  #   utils.shutdown_vm()


if __name__ == "__main__":
  app.run(main)
