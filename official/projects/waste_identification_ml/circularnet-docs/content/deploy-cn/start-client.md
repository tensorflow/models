# Run the prediction pipeline

Follow these steps to run the prediction pipeline and process your input files
on Google Cloud:

1. Open the `Client` directory in the `Triton_TF_Cloud_Deployment` directory:

    ```
    cd models/official/projects/waste_identification_ml/Triton_TF_Cloud_Deployment/client/
    ```

3. If you have to modify the scripts to provide your specific paths and values
   for the prediction pipeline, edit the corresponding parameter values on the
   script. The following example modifies the image pipeline script:

    ```
    vim run_images.sh
    ```

	The Vim editor displays the following parameters:

    ```
    --input_directory=<path-to-input-bucket>
    --output_directory=<path-to-output-bucket>
    --height=<height>
    --width=<width>
    --model=<circularnet-model>
    --score=<score>
    --search_range_x=<search-range>
    --search_range_y=<search-range>
    --memory=<memory>
    --project_id=<project-id>
    --bq_dataset_id=<bigquery-dataset-id>
    --bq_table_id=<bigquery-table-id>
    --overwrite=<overwrtie_table>
    --tracking_visualization=<visualize-tracking-results>
    --cropped_objects=<crop-objects-per-category>
    ```

    Replace the following:

    -  `<path-to-input-bucket>`: The path to [the Cloud Storage input bucket you
       created], for example `gs://my-input-bucket/`.
    -  `<path-to-output-bucket>`: The path to [the Cloud Storage output bucket
       you created], for example `gs://my-output-bucket/`.
    -  `<project-id>`: The ID of your Google Cloud project, for example,
       `my-project`.
    -  `<bigquery-dataset-id>`: The ID that you want to assign to a BigQuery \
        dataset to
       store prediction results, for example, `circularnet_dataset`.
    -  `<bigquery-table-id>`: The ID that you want to assign to a BigQuery \
        table to store
       prediction results, for example, `circularnet_table`. If the table
       already exists in your Google Cloud project, the pipeline appends results
       to that table.

    Save changes and exit the Vim editor. To do this, press the **Esc** key,
    type `:wq`, and then press **Enter**.

    Note : for creating cloud storage bucket follow [this guide](official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/prediction-pipeline-in-cloud.md#Create-the-Cloud-Storage-input-and-output-buckets)

4. Enter the `screen` session for the client:

    ```
    screen -R inference
    ```

    The `screen` session opens and displays the ongoing operations on the
    server. The models must show a `READY` status on the `screen` session when
    they are successfully deployed.

5. Run the prediction pipeline:

    ```
    bash run_images.sh
    ```

    **Note:** If you have a large amount of input files, you can run the
    pipeline in a `screen` session in the background without worrying about the
    terminal closing down. First, you launch the `screen` session with the
    `screen -R client` command. A new session shell launches. Then, run the
    `bash run_images.sh` script in the new shell.

6. If you want to exit the `screen` session without stopping the inference, press
   **Ctrl + A + D** keys.

The script also creates a `logs` folder inside the `client` folder that saves
the logs with the troubleshooting results and records from the models.

You have finished running the prediction pipeline and applying the prediction models to your files for further analysis. You can find the image results with the applied masks in your output bucket. You can also open the generated [BigQuery](https://cloud.google.com/bigquery) table to see the model analytics results and [preview table data](https://cloud.google.com/bigquery/docs/quickstarts/load-data-console#preview_table_data). To create new results, repeat the steps in this section every time you modify the files in your input bucket.

**Important:** If you rerun the prediction pipeline on the same video or image file, you must delete the results created the first time you ran the script from the output bucket to avoid conflicting issues. [Manage the lifecycle of objects in your Cloud Storage buckets](https://cloud.google.com/storage/docs/lifecycle) to help manage costs.