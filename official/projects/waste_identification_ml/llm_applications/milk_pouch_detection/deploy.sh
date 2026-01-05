#!/bin/bash

# ==============================================================================
# Fully Automated Deployment Script for Milk Pouch Detection.
# Deploys a Compute Engine service, along with the
# necessary GCP components, including:
# 1. GCP project configuration.
# 2. Service API enablement.
# 3. BigQuery dataset and table creation.
# 4. GCS bucket creation.
# 5. Artifact Registry repository creation.
# 6. Container image build and push (via Cloud Build).
# 7. Service account creation and permission granting.
# 8. Compute Engine deployment.
#
# This script is designed to be run locally.
# It requires the Google Cloud CLI (gcloud) to be installed and authenticated.
#
# Usage:
#   ./deploy.sh \
#     [--gcp_project_id=<project_id>] \
#     [--region=<region>] \
#     [--zone=<zone>] \
#     [--device=cpu|gpu] \
#     [--compute=gce] \
#     [--source_bucket_name=<source_bucket_name>]
#
# Arguments:
#   --gcp_project_id: Specify the GCP project ID.
#   --region: Specify the region for the resources. Default: asia-south1.
#   --zone: Specify the zone for the resources. Default: asia-south1-a.
#   --device: Specify the device type (cpu or gpu). Default: cpu.
#   --compute: Specify the compute platform (gce). Default: gce.
#   --source_bucket_name: Specify the source GCS bucket name.
#
# Example:
#   ./deploy.sh --device=gpu --compute=gce
#
# ==============================================================================

# If any command fails, the script will stop immediately
set -euo pipefail

# -----------------------------------------------------------------------------
# Configuration Block
# -----------------------------------------------------------------------------

# --- GCE Configuration ---
# Name of the GCE instance if using GCE for processing
export INSTANCE_NAME="milk-pouch-processor-vm"

# --- Script Configuration ---
# Name of the Artifact Registry repository
export REPO_NAME="milk-pouch-classification-repo"

# Name of the container image and Cloud Run service
export IMAGE_NAME="milk-pouch-classification-service"

# [Output] Name of the BigQuery Dataset
export BQ_DATASET="milk_pouch_classification"

# [Output] Name of the BigQuery Table
export BQ_TABLE="milk_pouch_classification_results"

# -----------------------------------------------------------------------------
# Script Logic - Do not modify the following
# -----------------------------------------------------------------------------

# --- Argument Parsing ---
# Set default values for device and compute platform
PROJECT_ID="project-id-placeholder"
REGION="asia-south1"
ZONE="asia-south1-a" # Zone for the GCE instance
DEVICE="gpu" # Default to GPU
COMPUTE="gce" # Default to gce
SOURCE_BUCKET_NAME=""

# Parse command-line arguments --device [cpu|gpu] and --compute [cloud-run|gce]
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --gcp_project_id) PROJECT_ID="$2"; shift ;;
    --region) REGION="$2"; shift ;;
    --zone) ZONE="$2"; shift ;;
    --device) DEVICE="$2"; shift ;;
    --compute) COMPUTE="$2"; shift ;;
    --source_bucket_name) SOURCE_BUCKET_NAME="$2"; shift ;;
    *) echo "Unknown parameter passed"; exit 1 ;;
  esac
  shift
done

# Validate Project ID
if [[ "${PROJECT_ID}" == "project-id-placeholder" ]]; then
  echo "❌ Project ID is not specified. Please provide a valid project ID."
  exit 1
fi

# Validate compute platform
if [[ "${COMPUTE}" != "gce" ]]; then
  echo "❌ Invalid service type specified. Currently only GCE is supported."
  exit 1
fi

# Validate device type
if [[ "${DEVICE}" != "cpu" && "${DEVICE}" != "gpu" ]]; then
  echo "❌ Invalid device specified. Choose 'cpu' or 'gpu'."
  exit 1
fi

# [Input] GCS Bucket name for uploading original images
if [[ -z "${SOURCE_BUCKET_NAME}" ]]; then
  export SOURCE_BUCKET_NAME="milk-pouch-classification-uploads-${PROJECT_ID}"
fi


echo "🚀 Starting deployment for a '${DEVICE}' configuration..."
echo ""

# -----------------------------------------------------------------------------
# Final Configuration Summary
# -----------------------------------------------------------------------------
echo "✅ Deployment script is going to run with the following configuration."
echo ""
echo "--------------------------------------------------"
echo "Configuration Summary:"
echo "--------------------------------------------------"
echo "Project ID: ${PROJECT_ID}"
echo "Region: ${REGION}"
echo "Compute Platform: ${COMPUTE}"
echo "Device: ${DEVICE}"
echo ""
echo "Artifact Registry Repo: ${REPO_NAME}"
echo "Image Name: ${IMAGE_NAME}"
echo ""
echo "Source GCS Bucket: ${SOURCE_BUCKET_NAME}"
echo ""
echo "BigQuery Dataset: ${BQ_DATASET}"
echo "BigQuery Table: ${BQ_TABLE}"

if [[ "$COMPUTE" == "gce" ]]; then
  echo ""
  echo "--- GCE Configuration ---"
  echo "Instance Name: ${INSTANCE_NAME}"
  echo "Zone: ${ZONE}"
fi
echo "--------------------------------------------------"

echo "✅ Step 1: Configure gcloud CLI..."
gcloud config set project "${PROJECT_ID}"
gcloud config set run/region "${REGION}"
echo "Project has been set to ${PROJECT_ID}, and region has been set to ${REGION}."
echo ""

# ---

echo "✅ Step 2: Enable required GCP services..."
gcloud services enable \
  run.googleapis.com \
  compute.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  bigquery.googleapis.com \
  pubsub.googleapis.com \
  cloudscheduler.googleapis.com \
  cloudresourcemanager.googleapis.com
echo "All APIs have been enabled."
echo ""

# ---

echo "✅ Step 3: Create BigQuery Dataset and Table..."
bq --location="${REGION}" mk --dataset "${PROJECT_ID}:${BQ_DATASET}" \
  || echo "Dataset '${BQ_DATASET}' already exists."
bq mk --table "${PROJECT_ID}:${BQ_DATASET}.${BQ_TABLE}" \
  ./src/milk_pouch_results_schema.json \
  || echo "Table '${BQ_TABLE}' already exists."
echo "BigQuery resources are ready."
echo ""

# ---

echo "✅ Step 4: Create GCS Buckets..."
gsutil mb \
  -p "${PROJECT_ID}" \
  -l "${REGION}" \
  -c standard \
  -b on "gs://${SOURCE_BUCKET_NAME}" \
  || echo "Source Bucket 'gs://${SOURCE_BUCKET_NAME}' already exists."
echo "GCS Buckets are ready."
echo ""

# ---

echo "✅ Step 5: Create Artifact Registry repository..."
gcloud artifacts repositories create "${REPO_NAME}" \
  --repository-format=docker \
  --location="${REGION}" \
  --description="Docker repository for ML models" \
  || echo "Repository '${REPO_NAME}' already exists."
echo "Artifact Registry repository is ready."
echo ""

# ---

echo "✅ Step 6: Build container image using Cloud Build (with cloudbuild.yaml)..."
gcloud builds submit --timeout=2h --config cloudbuild.yaml \
  --substitutions=_REGION="${REGION}",_REPO_NAME="${REPO_NAME}",_IMAGE_NAME="${IMAGE_NAME}",_GCS_PATH="gs://${SOURCE_BUCKET_NAME}"

echo "Skipping container build step. Assuming image already exists in Artifact Registry."
echo ""

# ---

echo "✅ Step 7: Create a dedicated service account and grant permissions..."
SERVICE_ACCOUNT_NAME="milk-pouch-classification-sa"
SERVICE_ACCOUNT_EMAIL="${SERVICE_ACCOUNT_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

# Create service account if it doesn't exist
gcloud iam service-accounts create "${SERVICE_ACCOUNT_NAME}" \
  --display-name="Service Account for ${IMAGE_NAME}" \
  || echo "Service account '${SERVICE_ACCOUNT_NAME}' already exists."

echo "Granting IAM permissions to service account..."
PROJECT_NUMBER=$( \
  gcloud projects describe "${PROJECT_ID}" --format="value(projectNumber)")

# Allow the service to invoke itself (required for Pub/Sub push
# subscriptions)
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/run.invoker" \
  --condition=None > /dev/null 2>&1

# Allow the service to write to GCS
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/storage.objectAdmin" \
  --condition=None > /dev/null 2>&1

# Allow the service to write to BigQuery
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/bigquery.dataEditor" \
  --condition=None > /dev/null 2>&1

# Allow the service to use Datastore
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/datastore.user" \
  --condition=None > /dev/null 2>&1

# Allow the service to write logs
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/logging.logWriter" \
  --condition=None > /dev/null 2>&1

# Allow the service to act as a Pub/Sub subscriber
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${SERVICE_ACCOUNT_EMAIL}" \
  --role="roles/pubsub.subscriber" \
  --condition=None > /dev/null 2>&1

# Allow GCS to publish messages to the Pub/Sub topic
GCS_SERVICE_AGENT="service-${PROJECT_NUMBER}@gs-project-accounts.iam.gserviceaccount.com"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
  --member="serviceAccount:${GCS_SERVICE_AGENT}" \
  --role="roles/pubsub.publisher" \
  --condition=None > /dev/null 2>&1

echo "All necessary IAM permissions have been granted."
echo ""

# ---

echo "✅ Step 8: Deploy Compute Service..."
if [[ "${COMPUTE}" == "gce" ]]; then
  # --- GCE Deployment ---
  echo "Setting GCP project..."
  gcloud config set project ${PROJECT_ID}

  echo "Deploying on the latest Deep Learning VM Image..."
  GCE_CREATE_CMD="gcloud compute instances create ${INSTANCE_NAME} \
    --project=${PROJECT_ID} \
    --zone=${ZONE} \
    --machine-type="n1-standard-4" \
    --image-family="common-cu128-ubuntu-2204-nvidia-570" \
    --image-project="deeplearning-platform-release" \
    --boot-disk-size=200GB \
    --scopes="cloud-platform" \
    --maintenance-policy=TERMINATE \
    --no-shielded-secure-boot \
    --metadata-from-file="startup-script=gce_startup.sh" \
    --metadata="IMAGE_URI=${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:latest""

  if [[ "${DEVICE}" == "gpu" ]]; then
    GCE_CREATE_CMD="${GCE_CREATE_CMD} --accelerator=\"type=nvidia-tesla-t4,count=1\""
  fi

  if eval "${GCE_CREATE_CMD}"; then
    echo ""
    echo "✅ Success! Compute Engine VM created."
  else
    GCE_EXIT_CODE=$?
    if gcloud compute instances describe "${INSTANCE_NAME}" --zone "${ZONE}" --project "${PROJECT_ID}" > /dev/null 2>&1; then
      echo "Compute Engine VM '${INSTANCE_NAME}' already exists."
    else
      echo "❌ Compute Engine VM creation failed with exit code ${GCE_EXIT_CODE}."
      echo "Use cmd: gcloud compute images list --project deeplearning-platform-release --no-standard-images --filter='family:common-cu128 AND ubuntu' to check the available images."
    fi
    exit 1
  fi

  echo ""
  echo "The process is now driven by the Compute Engine with '${IMAGE_NAME}'."
  echo ""
fi
echo ""

# ---

echo "🚀🚀🚀 Deployment complete! 🚀🚀🚀"
echo "You can upload files to the source bucket to be processed in the next run:"
echo "gsutil cp your-local-image.jpg gs://${SOURCE_BUCKET_NAME}/"
echo "Check the results in the bucket's subfolders."
