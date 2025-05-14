## Prepare your training dataset

Begin by capturing images with your camera and performing the necessary
preprocessing steps. Then, annotate the captured images to identify the
materials present in each one. These annotations allow the model to learn which
materials it must recognize during training. You must save annotations in COCO
JSON format.

The GitHub repository provides the required
[preprocessing scripts](https://github.com/tensorflow/models/tree/master/official/projects/waste_identification_ml/pre_processing)
to prepare the model for retraining. These scripts convert your annotated images
into [TFRecords](https://www.tensorflow.org/tutorials/load_data/tfrecord), the
required input format for TensorFlow models.

The preprocessing scripts of the repository let you perform the following
actions:

-   Convert image annotations to the COCO-annotated JSON file format, which
    is necessary for labels and metadata on a dataset.
-   Clean and prepare a COCO-annotated JSON file for training an ML model,
    ensuring that your dataset is clean, consistent, and ready for effective
    model training.
-   Filter out irrelevant or noisy annotations that could negatively impact
    the training process.
-   Adjust annotations to ensure they are in the optimal format for model
    training.
-   Verify that all annotated images exist and are not corrupted.
-   Merge COCO-annotated JSON files into a single file and convert it into
    TFRecord files, the required format for training with the Mask R-CNN model.

Before proceeding with the retraining pipeline, run the preprocessing scripts
with your data on your local workstation, remote server, or database. Once your
TFRecords are ready,
[upload them to a Cloud Storage bucket](https://cloud.google.com/storage/docs/uploading-objects).
The bucket can have two locations, one for the training dataset and another for
the validation dataset.

---

## Configure the training job

Configure your model training by customizing values in
[the `CircularNET_Vertex_AI_ReTraining_v1.ipynb` script](https://github.com/tensorflow/models/blob/master/official/projects/waste_identification_ml/model_retraining/CircularNET_Vertex_AI_ReTraining_v1.ipynb),
including your project ID, bucket URI, and region. Provide the following
information in the corresponding script variables:

-   `input_train_data_path`: path to the TFRecords of your training dataset
    in the Cloud Storage bucket.
-   `input_validation_data_path`: path to the TFRecords of your validation
    dataset in the Cloud Storage bucket.
-   `init_checkpoint_path`: path to the initial checkpoints with the model's
    weights. You can use the open-source initial checkpoints in this variable
    from
    [the configuration file](https://github.com/tensorflow/models/blob/master/official/projects/waste_identification_ml/model_retraining/config/config_v1.yaml).
-   `config_file_path`: path to the configuration file containing parameters
    for fine-tuning the training. You can use the open-source
    [configuration file](https://github.com/tensorflow/models/blob/master/official/projects/waste_identification_ml/model_retraining/config/config_v1.yaml)
    in this variable.
-   `service_account`: name of the Vertex AI service account with
    permissions to perform training jobs.

The script's placeholders, such as `PROJECT_ID`, `REGION`, and
`STAGING_BUCKET`, must be replaced with your Google Cloud project, region, and
Cloud Storage bucket, respectively.

**Note:** You can also customize other script variables, such as `num_classes`,
which refers to the number of categories for materials or other classes in your
annotated images.