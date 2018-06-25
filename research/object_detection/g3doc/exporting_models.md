# Exporting a trained model for inference

After your model has been trained, you should export it to a Tensorflow
graph proto. A checkpoint will typically consist of three files:

* model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001
* model.ckpt-${CHECKPOINT_NUMBER}.index
* model.ckpt-${CHECKPOINT_NUMBER}.meta

After you've identified a candidate checkpoint to export, run the following
command from tensorflow/models/research:

``` bash
# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory ${EXPORT_DIR}
```

Afterwards, you should see the directory ${EXPORT_DIR} containing the following:

* output_inference_graph.pb, the frozen graph format of the exported model
* saved_model/, a directory containing the saved model format of the exported model
* model.ckpt.*, the model checkpoints used for exporting
* checkpoint, a file specifying to restore included checkpoint files

# Exporting a trained model for tflite inference

After your model has been trained, you should export it to a Tensorflow
graph proto. A checkpoint will typically consist of three files:

* model.ckpt-${CHECKPOINT_NUMBER}.data-00000-of-00001
* model.ckpt-${CHECKPOINT_NUMBER}.index
* model.ckpt-${CHECKPOINT_NUMBER}.meta

After you've identified a candidate checkpoint to export, run the following
command from tensorflow/models/research:

``` bash
# From tensorflow/models/research/
python object_detection/export_inference_graph.py \
    --input_type image_tensor \
     --data_type=float \
    --pipeline_config_path ${PIPELINE_CONFIG_PATH} \
    --trained_checkpoint_prefix ${TRAIN_PATH} \
    --output_directory ${EXPORT_DIR} \
    --input_shape -1,${HEIGHT},${WIDTH},3 \
    --no_preprocess --no_postprocess
```
Afterwards, you should see the directory ${EXPORT_DIR} containing the following:

* output_inference_graph.pb, the frozen graph format of the exported model
* saved_model/, a directory containing the saved model format of the exported model
* model.ckpt.*, the model checkpoints used for exporting
* checkpoint, a file specifying to restore included checkpoint files

Finally, you should use toco tool to convert .pb fronzen graph file to .tflite format.
If you do not have toco tool yet, Please install bazel build tool, then clone the tensorflow repo.
``` bash
git clone https://github.com/tensorflow/tensorflow
cd tensorflow
bazel build tensorflow/contrib/lite/toco:toco
``` 
For float version:
``` bash
bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=${EXPORT_DIR}/frozen_inference_graph.pb \
  --output_file=${EXPORT_DIR}/tflite_model.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=FLOAT \
  --input_shapes="1,${HEIGHT},${WIDTH},3" \
  --input_arrays=image_tensor \
  --output_arrays="box_encodings,class_predictions_with_background" \
  --std_values=127.5 --mean_values=127.5 \
```
For quantize version:
``` bash
bazel-bin/tensorflow/contrib/lite/toco/toco \
  --input_file=${EXPORT_DIR}/frozen_inference_graph.pb \
  --output_file=${EXPORT_DIR}/tflite_model_quantize.tflite \
  --input_format=TENSORFLOW_GRAPHDEF --output_format=TFLITE \
  --inference_type=QUANTIZED_UINT8 \
  --input_shapes="1,${HEIGHT},${WIDTH},3" \
  --input_arrays=image_tensor \
  --output_arrays="box_encodings,class_predictions_with_background" \
  --std_values=127.5 --mean_values=127.5
```
<br>For further information you can visit:</br>
<br>[Fixed Point Quantization](https://www.tensorflow.org/performance/quantization)</br>
<br>[Android Demo App Intro](https://www.tensorflow.org/mobile/tflite/demo_android)</br>
<br>[Android Demo App Repo](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/examples/android)</br>