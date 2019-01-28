#!/bin/bash
set -e

# Example settings:
# export TPU="taylorrobie-tpu-0"
# export BUCKET="gs://taylorrobie-tpu-test-bucket-2"

# Remove IDE "not assigned" warning highlights.
TPU=${TPU:-""}
BUCKET=${BUCKET:-""}

if [[ -z ${TPU} ]]; then
  echo "Please set 'TPU' to the name of the TPU to be used."
  exit 1
fi

if [[ -z ${BUCKET} ]]; then
  echo "Please set 'BUCKET' to the GCS bucket to be used."
  exit 1
fi

./run.sh
