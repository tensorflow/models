The script that runs [the prediction pipeline](/official/projects/waste_identification_ml/circularnet-docs/content/learn-about-pipeline) on an
edge device applies the prediction models to analyze images.

After [setting up a server](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/start-server) in your edge device, you
can start recording videos of objects passing on your conveyor belt to gather
data for analysis and storing files locally from [the camera](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/). The next step is transferring those
videos or image files to a folder in the edge device, where the prediction
pipeline processes the images.

The results of each video or image prediction are also stored locally in an
output folder in the edge device, ensuring efficient data management. After
systematically processing each file in your local directory, the pipeline
creates an output directory with results for further use and analysis.

This page explains how to run the prediction pipeline to apply the models to
images stored locally in the edge device. You can then manage data according to
your needs, such as exporting the results to BigQuery tables and connecting a
visualization dashboard to [display results as charts and reports](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/).<br/><br/>

{{< table_of_contents >}}

---

## Run the prediction pipeline

Follow these steps to run the prediction pipeline and process your input files
in an edge device:

1. Open the terminal of your edge device to interact with the operating system
   through the command line.
1. [Start the server](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/start-server).
1. Display the names of the models you loaded to the Triton inference server:

    ```
    cat triton_server.sh
    ```

    The first lines of the output show the names of the loaded models in square
    brackets. You can call any of these models when running the prediction
    pipeline.

    **Important:** Run the previous command in the `server` folder, which
    contains the `triton_server.sh` script.

1. Exit the `server` folder and open the `client` folder in the
   `prediction_pipeline` directory:

    ```
    cd ..
    cd client/
    ```

    This folder contains the `pipeline_images.py` Python file that stores the
    complete prediction pipeline for input images. The `run_edge_images.sh`
    script runs this Python file automatically.

1. If you have to modify the script to provide your specific paths and values
   for the prediction pipeline, edit the corresponding parameter values on the
   script. The following example modifies the image pipeline script:

    ```
    vim run_edge_images.sh
    ```

    The Vim editor displays the following parameters:

    ```
    --input_directoy=<path-to-input-folder>
    --output_directory=<path-to-output-folder> height=<height> width=<width>
    --material_model=<material-model> material_form_model=<material-form-model>
    --score=<score> search_range=<search-range> memory=<memory>
    ```

    Replace the following:

    -  `<path-to-input-folder>`: The path to the local folder for input images
       in the edge device, for example `/home/images/input_files/`.
    -  `<path-to-output-folder>`: The path to the local folder for output image
       results in the edge device, for example `/home/images/output_files/`.
    -  `<height>`: The height in pixels of the image or video frames that the
       model expects for prediction, for example, 512.
    -  `<width>`: The width in pixels of the image or video frames that the
       model expects for prediction, for example, 1024.
    -  `<material-model>`: The name of the material model in the Triton
       inference server that you want to call, for example,
       `material_resnet_v2_512_1024`.
    -  `<material-form-model>`: The name of the material form model in the
       Triton inference server that you want to call, for example,
       `material_form_resnet_v2_512_1024`.
    -  `<score>`: The threshold for model prediction, for example, 0.40.
    -  `<search-range>`: The pixels up to which you want to track an object for
       object tracking in consecutive frames, for example, 100.
    -  `<memory>`: The frames up to which you want to track an object, for
       example, 20.

    Save changes and exit the Vim editor. To do this, press the **Esc** key,
    type `:wq`, and then press **Enter**.

1. Run the prediction pipeline:

    ```
    bash run_edge_images.sh
    ```

The script also creates a `logs` folder inside the `client` folder that saves
the logs with the troubleshooting results and records from the models.

You have finished running the prediction pipeline and applying the prediction
models to your files for further analysis. You can find the image results with
the applied masks in your output folder in the edge device. You can also export
your results manually to a [BigQuery](https://cloud.google.com/bigquery) table
to connect it with a visualization dashboard for [data analysis and reporting](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/).

**Important:** If you rerun the prediction pipeline on the same file, you must
delete the results created the first time you ran the script from the output
folder to avoid conflicting issues.

## What's next

-  [View data analysis and reporting](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/)