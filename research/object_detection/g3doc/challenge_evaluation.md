# Open Images Challenge Evaluation

The Object Detection API is currently supporting several evaluation metrics used in the [Open Images Challenge 2018](https://storage.googleapis.com/openimages/web/challenge.html).
In addition, several data processing tools are available. Detailed instructions on using the tools for each track are available below.

**NOTE**: links to the external website in this tutorial may change after the Open Images Challenge 2018 is finished.

## Object Detection Track

The [Object Detection metric](https://storage.googleapis.com/openimages/web/object_detection_metric.html) protocol requires a pre-processing of the released data to ensure correct evaluation. The released data contains only leaf-most bounding box annotations and image-level labels.
The evaluation metric implementation is available in the class `OpenImagesDetectionChallengeEvaluator`.

1. Download class hierarchy of Open Images Challenge 2018 in JSON format from [here](https://storage.googleapis.com/openimages/challenge_2018/bbox_labels_500_hierarchy.json).
2. Download ground-truth [boundling boxes](https://storage.googleapis.com/openimages/challenge_2018/train/challenge-2018-train-annotations-bbox.csv) and [image-level labels](https://storage.googleapis.com/openimages/challenge_2018/train/challenge-2018-train-annotations-human-imagelabels.csv).
3. Filter the rows corresponding to the validation set images you want to use and store the results in the same CSV format.
4. Run the following command to create hierarchical expansion of the bounding boxes annotations:

```
HIERARCHY_FILE=/path/to/bbox_labels_500_hierarchy.json
BOUNDING_BOXES=/path/to/challenge-2018-train-annotations-bbox
IMAGE_LABELS=/path/to/challenge-2018-train-annotations-human-imagelabels

python object_detection/dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${BOUNDING_BOXES}.csv \
    --output_annotations=${BOUNDING_BOXES}_expanded.csv \
    --annotation_type=1

python object_detection/dataset_tools/oid_hierarchical_labels_expansion.py \
    --json_hierarchy_file=${HIERARCHY_FILE} \
    --input_annotations=${IMAGE_LABELS}.csv \
    --output_annotations=${IMAGE_LABELS}_expanded.csv \
    --annotation_type=2
```

After step 4 you will have produced the ground-truth files suitable for running 'OID Challenge Object Detection Metric 2018' evaluation.

```
INPUT_PREDICTIONS=/path/to/detection_predictions.csv
OUTPUT_METRICS=/path/to/output/metrics/file

python models/research/object_detection/metrics/oid_od_challenge_evaluation.py \
    --input_annotations_boxes=${BOUNDING_BOXES}_expanded.csv \
    --input_annotations_labels=${IMAGE_LABELS}_expanded.csv \
    --input_class_labelmap=object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt \
    --input_predictions=${INPUT_PREDICTIONS} \
    --output_metrics=${OUTPUT_METRICS} \
```

### Running evaluation on CSV files directly

5. If you are not using Tensorflow, you can run evaluation directly using your algorithm's output and generated ground-truth files. {value=5}


### Running evaluation using TF Object Detection API

5. Produce tf.Example files suitable for running inference: {value=5}

```
RAW_IMAGES_DIR=/path/to/raw_images_location
OUTPUT_DIR=/path/to/output_tfrecords

python object_detection/dataset_tools/create_oid_tf_record.py \
    --input_box_annotations_csv ${BOUNDING_BOXES}_expanded.csv \
    --input_image_label_annotations_csv ${IMAGE_LABELS}_expanded.csv \
    --input_images_directory ${RAW_IMAGES_DIR} \
    --input_label_map object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt \
    --output_tf_record_path_prefix ${OUTPUT_DIR} \
    --num_shards=100
```

6. Run inference of your model and fill corresponding fields in tf.Example: see [this tutorial](object_detection/g3doc/oid_inference_and_evaluation.md) on running the inference with Tensorflow Object Detection API models. {value=6}

7. Finally, run the evaluation script to produce the final evaluation result.

```
INPUT_TFRECORDS_WITH_DETECTIONS=/path/to/tf_records_with_detections
OUTPUT_CONFIG_DIR=/path/to/configs

echo "
label_map_path: 'object_detection/data/oid_object_detection_challenge_500_label_map.pbtxt'
tf_record_input_reader: { input_path: '${INPUT_TFRECORDS_WITH_DETECTIONS}' }
" > ${OUTPUT_CONFIG_DIR}/input_config.pbtxt

echo "
metrics_set: 'oid_challenge_detection_metrics'
" > ${OUTPUT_CONFIG_DIR}/eval_config.pbtxt

OUTPUT_METRICS_DIR=/path/to/metrics_csv

python object_detection/metrics/offline_eval_map_corloc.py \
    --eval_dir=${OUTPUT_METRICS_DIR} \
    --eval_config_path=${OUTPUT_CONFIG_DIR}/eval_config.pbtxt \
    --input_config_path=${OUTPUT_CONFIG_DIR}/input_config.pbtxt
```

The result of the evaluation will be stored in `${OUTPUT_METRICS_DIR}/metrics.csv`

For the Object Detection Track, the participants will be ranked on:

- "OpenImagesChallenge2018_Precision/mAP@0.5IOU"

## Visual Relationships Detection Track

The [Visual Relationships Detection metrics](https://storage.googleapis.com/openimages/web/vrd_detection_metric.html) can be directly evaluated using the ground-truth data and model predictions. The evaluation metric implementation is available in the class `VRDRelationDetectionEvaluator`,`VRDPhraseDetectionEvaluator`.

1. Download the ground-truth [visual relationships annotations](https://storage.googleapis.com/openimages/challenge_2018/train/challenge-2018-train-vrd.csv) and [image-level labels](https://storage.googleapis.com/openimages/challenge_2018/train/challenge-2018-train-vrd-labels.csv).
2. Filter the rows corresponding to the validation set images you want to use and store the results in the same CSV format.
3. Run the follwing command to produce final metrics:

```
INPUT_ANNOTATIONS_BOXES=/path/to/challenge-2018-train-vrd.csv
INPUT_ANNOTATIONS_LABELS=/path/to/challenge-2018-train-vrd-labels.csv
INPUT_PREDICTIONS=/path/to/predictions.csv
INPUT_CLASS_LABELMAP=/path/to/oid_object_detection_challenge_500_label_map.pbtxt
INPUT_RELATIONSHIP_LABELMAP=/path/to/relationships_labelmap.pbtxt
OUTPUT_METRICS=/path/to/output/metrics/file

echo "item { name: '/m/02gy9n' id: 602 display_name: 'Transparent' }
item { name: '/m/05z87' id: 603 display_name: 'Plastic' }
item { name: '/m/0dnr7' id: 604 display_name: '(made of)Textile' }
item { name: '/m/04lbp' id: 605 display_name: '(made of)Leather' }
item { name: '/m/083vt' id: 606 display_name: 'Wooden'}
">>${INPUT_CLASS_LABELMAP}

echo "item { name: 'at' id: 1 display_name: 'at' }
item { name: 'on' id: 2 display_name: 'on (top of)' }
item { name: 'holds' id: 3 display_name: 'holds' }
item { name: 'plays' id: 4 display_name: 'plays' }
item { name: 'interacts_with' id: 5 display_name: 'interacts with' }
item { name: 'wears' id: 6 display_name: 'wears' }
item { name: 'is' id: 7 display_name: 'is' }
item { name: 'inside_of' id: 8 display_name: 'inside of' }
item { name: 'under' id: 9 display_name: 'under' }
item { name: 'hits' id: 10 display_name: 'hits' }
"> ${INPUT_RELATIONSHIP_LABELMAP}

python object_detection/metrics/oid_vrd_challenge_evaluation.py \
    --input_annotations_boxes=${INPUT_ANNOTATIONS_BOXES} \
    --input_annotations_labels=${INPUT_ANNOTATIONS_LABELS} \
    --input_predictions=${INPUT_PREDICTIONS} \
    --input_class_labelmap=${INPUT_CLASS_LABELMAP} \
    --input_relationship_labelmap=${INPUT_RELATIONSHIP_LABELMAP} \
    --output_metrics=${OUTPUT_METRICS}
```

The participants of the challenge will be evaluated by weighted average of the following three metrics:

- "VRDMetric_Relationships_mAP@0.5IOU"
- "VRDMetric_Relationships_Recall@50@0.5IOU"
- "VRDMetric_Phrases_mAP@0.5IOU"
