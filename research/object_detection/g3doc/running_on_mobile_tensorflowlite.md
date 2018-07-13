# Running on mobile with TensorFlow Lite

In this section, we will show you how to use [TensorFlow
Lite](https://www.tensorflow.org/mobile/tflite/) to get a smaller model and
allow you take advantage of ops that have been optimized for mobile devices.
TensorFlow Lite is TensorFlow’s lightweight solution for mobile and embedded
devices. It enables on-device machine learning inference with low latency and a
small binary size. TensorFlow Lite uses many techniques for this such as
quantized kernels that allow smaller and faster (fixed-point math) models.

For this section, you will need to build [TensorFlow from
source](https://www.tensorflow.org/install/install_sources) to get the
TensorFlow Lite support for the SSD model. You will also need to install the
[bazel build
tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#bazel).

To make these commands easier to run, let’s set up some environment variables:

```shell
export CONFIG_FILE=PATH_TO_BE_CONFIGURED/pipeline.config
export CHECKPOINT_PATH=PATH_TO_BE_CONFIGURED/model.ckpt
export OUTPUT_DIR=/tmp/tflite
```

We start with a checkpoint and get a TensorFlow frozen graph with compatible ops
that we can use with TensorFlow Lite. First, you’ll need to install these
[python
libraries](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md).
Then to get the frozen graph, run the export_tflite_ssd_graph.py script from the
`models/research` directory with this command:

```shell
object_detection/export_tflite_ssd_graph.py \
--pipeline_config_path=$CONFIG_FILE \
--trained_checkpoint_prefix=$CHECKPOINT_PATH \
--output_directory=$OUTPUT_DIR \
--add_postprocessing_op=true
```

In the /tmp/tflite directory, you should now see two files: tflite_graph.pb and
tflite_graph.pbtxt. Note that the add_postprocessing flag enables the model to
take advantage of a custom optimized detection post-processing operation which
can be thought of as a replacement for
[tf.image.non_max_suppression](https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression).
Make sure not to confuse export_tflite_ssd_graph with export_inference_graph in
the same directory. Both scripts output frozen graphs: export_tflite_ssd_graph
will output the frozen graph that we can input to TensorFlow Lite directly and
is the one we’ll be using.

Next we’ll use TensorFlow Lite to get the optimized model by using
[TOCO](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/toco),
the TensorFlow Lite Optimizing Converter. This will convert the resulting frozen
graph (tflite_graph.pb) to the TensorFlow Lite flatbuffer format (detect.tflite)
via the following command. For a quantized model, run this from the tensorflow/
directory:

```shell
bazel run --config=opt tensorflow/contrib/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3' \
--inference_type=QUANTIZED_UINT8 \
--mean_values=128 \
--std_values=128 \
--change_concat_input_ranges=false \
--allow_custom_ops
```

This command takes the input tensor normalized_input_image_tensor after resizing
each camera image frame to 300x300 pixels. The outputs of the quantized model
are named 'TFLite_Detection_PostProcess', 'TFLite_Detection_PostProcess:1',
'TFLite_Detection_PostProcess:2', and 'TFLite_Detection_PostProcess:3' and
represent four arrays: detection_boxes, detection_classes, detection_scores, and
num_detections. The documentation for other flags used in this command is
[here](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/lite/toco/g3doc/cmdline_reference.md).
If things ran successfully, you should now see a third file in the /tmp/tflite
directory called detect.tflite. This file contains the graph and all model
parameters and can be run via the TensorFlow Lite interpreter on the Android
device. For a floating point model, run this from the tensorflow/ directory:

```shell
bazel run --config=opt tensorflow/contrib/lite/toco:toco -- \
--input_file=$OUTPUT_DIR/tflite_graph.pb \
--output_file=$OUTPUT_DIR/detect.tflite \
--input_shapes=1,300,300,3 \
--input_arrays=normalized_input_image_tensor \
--output_arrays='TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3'  \
--inference_type=FLOAT \
--allow_custom_ops
```

# Running our model on Android

To run our TensorFlow Lite model on device, we will need to install the Android
NDK and SDK. The current recommended Android NDK version is 14b and can be found
on the [NDK
Archives](https://developer.android.com/ndk/downloads/older_releases.html#ndk-14b-downloads)
page. Android SDK and build tools can be [downloaded
separately](https://developer.android.com/tools/revisions/build-tools.html) or
used as part of [Android
Studio](https://developer.android.com/studio/index.html). To build the
TensorFlow Lite Android demo, build tools require API >= 23 (but it will run on
devices with API >= 21). Additional details are available on the [TensorFlow
Lite Android App
page](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/contrib/lite/java/demo/README.md).

Next we need to point the app to our new detect.tflite file and give it the
names of our new labels. Specifically, we will copy our TensorFlow Lite
flatbuffer to the app assets directory with the following command:

```shell
cp /tmp/tflite/detect.tflite \
//tensorflow/contrib/lite/examples/android/app/src/main/assets
```

You will also need to copy your new labelmap labels_list.txt to the assets
directory.

We will now edit the BUILD file to point to this new model. First, open the
BUILD file tensorflow/contrib/lite/examples/android/BUILD. Then find the assets
section, and replace the line “@tflite_mobilenet_ssd_quant//:detect.tflite”
(which by default points to a COCO pretrained model) with the path to your new
TFLite model
“//tensorflow/contrib/lite/examples/android/app/src/main/assets:detect.tflite”.
Finally, change the last line in assets section to use the new label map as
well.

We will also need to tell our app to use the new label map. In order to do this,
open up the
tensorflow/contrib/lite/examples/android/app/src/main/java/org/tensorflow/demo/DetectorActivity.java
file in a text editor and find the definition of TF_OD_API_LABELS_FILE. Update
this path to point to your new label map file:
"file:///android_asset/labels_list.txt". Note that if your model is quantized,
the flag TF_OD_API_IS_QUANTIZED is set to true, and if your model is floating
point, the flag TF_OD_API_IS_QUANTIZED is set to false. This new section of
DetectorActivity.java should now look as follows for a quantized model:

```shell
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labels_list.txt";
```

Once you’ve copied the TensorFlow Lite file and edited your BUILD and
DetectorActivity.java files, you can build the demo app, run this bazel command
from the tensorflow directory:

```shell
 bazel build -c opt --config=android_arm{,64} --cxxopt='--std=c++11'
"//tensorflow/contrib/lite/examples/android:tflite_demo"
```

Now install the demo on a
[debug-enabled](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android#install)
Android phone via [Android Debug
Bridge](https://developer.android.com/studio/command-line/adb) (adb):

```shell
adb install bazel-bin/tensorflow/contrib/lite/examples/android/tflite_demo.apk
```
