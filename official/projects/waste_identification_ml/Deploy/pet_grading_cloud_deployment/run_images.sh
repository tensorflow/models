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
 --model_name              : Name of the model to download and use for inference.
 --threshold               : Confidence threshold for detections during
                             inference.
 --project_id              : Google Cloud Project ID for BigQuery operations.
 --bq_dataset_id           : BigQuery Dataset ID where results will be stored.
 --bq_table_id             : BigQuery Table ID where results will be stored.
 --overwrite               : If set to True, overwrites the pre-existing
                             BigQuery table.
EOF
#Activate the virtual environment
source myenv/bin/activate
# Check if the virtual environment is activated
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "Virtual environment 'myenv' activated successfully."
else
    echo "Failed to activate virtual environment. Exiting."
    exit 1
fi

python inference_pipeline.py \
	--input_directory=gs://circularnet_data/tmp/pet_pipeline_test \
	--output_directory=gs://circularnet_data/tmp/pet_pipeline_test \
	--project_id=waste-identification-ml-330916 \
	--bq_dataset_id=pet_dataset \
	--bq_table_id=pet_pipeline_test \
	--overwrite=True
