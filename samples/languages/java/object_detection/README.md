# Object Detection in Java

Example of using pre-trained models of the [TensorFlow Object Detection
API](https://github.com/tensorflow/models/tree/master/research/object_detection)
in Java.

## Quickstart

1. Download some metadata files:
   ```
   ./download.sh
   ```

2. Download a model from the [object detection API model
   zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).
   For example:
   ```
   mkdir -p models
   curl -L \
    http://download.tensorflow.org/models/object_detection/ssd_inception_v2_coco_2017_11_17.tar.gz \
   | tar -xz -C models/
   ```

3. Have some test images handy. For example:
   ```
   mkdir -p images
   curl -L -o images/test.jpg \
     https://pixnio.com/free-images/people/mother-father-and-children-washing-dog-labrador-retriever-outside-in-the-fresh-air-725x483.jpg
   ```

4. Compile and run!
   ```
   mvn -q compile exec:java \
     -Dexec.args="models/ssd_inception_v2_coco_2017_11_17/saved_model labels/mscoco_label_map.pbtxt images/test.jpg"
   ```

## Notes

- This example demonstrates the use of the TensorFlow [SavedModel
  format](https://www.tensorflow.org/guide/saved_model). If you have
  TensorFlow for Python installed, you could explore the model to get the names
  of the tensors using `saved_model_cli` command. For example:
  ```
  saved_model_cli show --dir models/ssd_inception_v2_coco_2017_11_17/saved_model/ --all
  ```

- The file in `src/main/object_detection/protos/` was generated using:

  ```
  ./download.sh
  protoc -Isrc/main/protobuf --java_out=src/main/java src/main/protobuf/string_int_label_map.proto
  ```

  Where `protoc` was downloaded from
  https://github.com/google/protobuf/releases/tag/v3.5.1
