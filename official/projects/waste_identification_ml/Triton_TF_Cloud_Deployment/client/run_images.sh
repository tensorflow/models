#!/bin/bash

cat << EOF
This script automates the execution of an Circularnet pipeline for image
processing.

Steps Performed:
 1. Activates the Python virtual environment named 'myenv'.
 2. Validates successful activation of the virtual environment.
 3. Executes the 'pipeline_images.py' script with the following parameters:

 Parameters:
 --input_directory         : GCS directory where the input images are stored for
                             inference.
 --output_directory        : GCS directory where the model inference outputs will be
                             saved.
 --height                  : Height to which input images are resized for the Mask
                             R-CNN model.
 --width                   : Width to which input images are resized for the Mask
                             R-CNN model.
 --model                   : Name of the model to download and use for inference.
 --score                   : Confidence threshold for detections during
                             inference.
 --search_range_x          : Max pixel movement allowed in the X direction for
                             object tracking between missed frames.
 --search_range_y          : Max pixel movement allowed in the Y direction for
                             object tracking between missed frames.
  --memory                 : Number of frames an object can be missed and still
                             be tracked.
  --project_id             : Google Cloud Project ID for BigQuery operations.
  --bq_dataset_id          : BigQuery Dataset ID where results will be stored.
  --bq_table_id            : BigQuery Table ID where results will be stored.
  --overwrite              : If set to True, overwrites the pre-existing
                             BigQuery table.
  --tracking_visualization : If set to True, visualizes the tracking results
                             from the tracking algorithm.
  --cropped_objects        : If set to True, crops the objects per category
                             according to the prediction and tracking results.
EOF

# Activate the virtual environment
source myenv/bin/activate

# Check if the virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment 'myenv' activated successfully."
else
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

python inference_pipeline.py \
	--input_directory=gs://recykal/TestData/Delterra \
	--output_directory=gs://recykal/TestData/output \
	--height=1024 \
	--width=1024 \
	--model=Jan2025_ver2_merged_1024_1024 \
	--score=0.70 \
	--search_range_x=150 \
	--search_range_y=20 \
	--memory=10  \
	--project_id=waste-identification-ml-330916 \
	--bq_dataset_id=circularnet_dataset \
	--bq_table_id=circularnet_table \
	--overwrite=True \
    --tracking_visualization=False \
	--cropped_objects=False