## Launch the training job

Run
[the `CircularNET_Vertex_AI_ReTraining_v1.ipynb` script](https://github.com/tensorflow/models/blob/master/official/projects/waste_identification_ml/model_retraining/CircularNET_Vertex_AI_ReTraining_v1.ipynb)
to configure parameters, launch the training job on Vertex AI, and start the
training.

**Note:** Wait to export the checkpoints to a saved TF model until the script
finishes running. The training job might take several days to complete. Vertex
AI manages the process in the background, so you don't need to keep your
computer running.

Follow these steps to monitor the training progress:

1.  Log in to your Google Cloud account.
1.  Navigate to Vertex AI.
1.  From the Vertex AI menu, click **Training**.
1.  On the **Training** page, open the **Hyperparameter tuning jobs** tab
    and select the name of your training job.
1.  To observe data about the training job, perform one of the following
    actions:
    -   Click **Open TensorBoard** to view detailed training metrics.
    -   Click **View Logs** to open the log console and monitor the
        detailed log messages.

---

## Export and test the model

After training is complete, you can export the model in various formats, such
as TensorFlow, for future use when you deploy the model.
[The `CircularNET_Vertex_AI_ReTraining_v1.ipynb` script](https://github.com/tensorflow/models/blob/master/official/projects/waste_identification_ml/model_retraining/CircularNET_Vertex_AI_ReTraining_v1.ipynb)
guides you through this process.

Additionally, you can test the model locally and run inferences on images to
validate its performance. The script also contains information about running
local tests.

Finally, deploy the exported model using an edge device or Google Cloud,
depending on your deployment requirements.