Before visualizing your prediction results on a dashboard, you must follow these
steps:

1. [Create a Google Cloud account](https://console.cloud.google.com/).
1. [Open the Google Cloud console](https://cloud.google.com/cloud-console).
1. Store your model results in BigQuery.
    -  If you [run the prediction pipeline for object tracking in Google Cloud](/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/prediction-pipeline-in-cloud),
       you automatically load the data to BigQuery, so no action is required.
    -  If you use an edge device or any other solution for object tracking
       outside of Google Cloud, you must [manually load the data from your
       database to
       BigQuery](https://cloud.google.com/bigquery/docs/loading-data).

**Tip:** You can push data from an edge device to the cloud by configuring cloud API access from your code running on the edge device. For information on Jetson NVIDIA devices, see their [Reference Cloud documentation](https://docs.nvidia.com/moj/cloud/cloud-overview.html).