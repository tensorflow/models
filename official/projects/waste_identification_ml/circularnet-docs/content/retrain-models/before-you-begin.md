## Before you begin

Before starting the retraining process, ensure you meet the following
requirements:

1.  [Get access to Google Cloud](https://console.cloud.google.com/).
1.  [Open the Google Cloud console](https://cloud.google.com/cloud-console).
1.  [Create a project on your Google Cloud account](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
1.  Enable the Vertex AI and Cloud Storage APIs to manage programmatic
    access and authentication.

    To enable APIs, see
    [Enabling an API in your Google Cloud project](https://cloud.google.com/endpoints/docs/openapi/enable-api).

1.  [Create a Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets)
    to store files.
1.  Allocate at least four GPUs for the training job on Vertex AI. For more
    information, see
    [Configure compute resources for custom training](https://cloud.google.com/vertex-ai/docs/training/configure-compute).
1.  Set up a service account for Vertex AI, with permissions to perform
    training jobs. For more information, see
    [Use a custom service account](https://cloud.google.com/vertex-ai/docs/general/custom-service-account).