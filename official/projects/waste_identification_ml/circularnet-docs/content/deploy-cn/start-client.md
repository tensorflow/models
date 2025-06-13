# Run the inference pipeline

Follow these steps to leverage the triton inference server you
[started earlier](start-server.md) to run the inference pipeline on your images.

Go to the `client` folder within the the `waste_identification_ml` project:

```bash
cd models/official/projects/waste_identification_ml/Triton_TF_Cloud_Deployment/client/
```

The inference pipeline uses a script to set various parameters for inference,
post-processing, and subsequent data analytics. We'll need to adjust some of
these:

```bash
vim run_images.sh
```

The Vim editor displays all pipeline parameters. The script contains
documentation for each parameter, but at minimum you will need to change:

```python
--input_directory=<path-to-input-bucket>

# This should be the path to the input bucket you created containing your
# images, e.g. gs://bucket/input-images
```

```python
--output_directory=<path-to-output-bucket>

# Like input, this should be the gcs bucket path to where you want the images
# with predictions to write to.
```

```python
--project_id=<project-id>

# The ID of your Google Cloud project housing your gcs bucket, for example,
# `my-gcp-project`.
```

```python
--bq_table_id=<bigquery-table-id>

'''
The ID that you want to use for your BigQuery table storing inference
results, e.g `circularnet_table`. If the table already exists BigQuery
within your project, the pipeline will either overwrite or append results,
depending on how you set the `overwrite` parameter
'''
```

Save changes and exit the Vim editor. To do this, press the **Esc** key, then
type `:wq`, and press **Enter**.

**Note:** For creating cloud storage bucket and adding images, follow
[this guide](https://github.com/tensorflow/models/blob/master/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/prediction-pipeline-in-cloud.md#Create-the-Cloud-Storage-input-and-output-buckets)

Next, enter a `screen` session for the inference client:

```bash
screen -R inference
```

Now run the inference pipeline:

```bash
bash run_images.sh
```

If you want to exit the screen session without stopping inference, press
**Ctrl + a** then **d** to detach from the screen session.

The script also creates a `logs` folder inside the `client` folder that saves
the logs with the troubleshooting results and records from the models.

Congratulations, you have finished running the inference pipeline!. You can find
individual inference results as images with overlaid object predictions in your
output bucket. You can also open the generated BigQuery table to see overall
analytics across all your images. You'll want to navigate to BigQuery within
your cloud project, find the appropriate table, and
[preview table data](https://cloud.google.com/bigquery/docs/quickstarts/load-data-console#preview_table_data).

**Important:** If you rerun the prediction pipeline on the same images, you
should delete any existing image results (output-bucket) created previously.
Also, see
[Manage the lifecycle of objects in your Cloud Storage buckets](https://cloud.google.com/storage/docs/lifecycle) to help manage image storage costs.