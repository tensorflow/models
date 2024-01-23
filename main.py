import os,sys
import numpy as np
import tensorflow as tf
sys.path.append('/home/runner/work/BboxSuggestion/BboxSuggestion/models')
sys.path.append('/home/runner/work/BboxSuggestion/BboxSuggestion/models/research')
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
import utils
import cv2, glob, shutil
from s3.s3_utils import s3_utils
from typing import Any

def inference(input_dir = 'images', output_dir = 'output',  ext = ['jpg', 'png', 'bmp', 'jpeg']):
    # 入力ディレクトリの存在確認
    if not os.path.isdir(input_dir): return

    # 出力ディレクトリの作成
    if os.path.isdir(output_dir): shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    #EfficientDet D0チェックポイントをダウンロード
    obj_det_path = '/worker/BboxSuggestion/research/object_detection'
    if not os.path.isdir(f'{obj_det_path}/test_data/'):
        utils.download_ckpt()
    pipeline_config = f'{obj_det_path}/test_data/efficientdet_d0_coco17_tpu-32/pipeline.config'
    model_dir = f'{obj_det_path}/test_data/efficientdet_d0_coco17_tpu-32/checkpoint'
    if not os.path.isfile(pipeline_config): assert False

    # label_mapへのパス文字列を置換
    utils.replace(pipeline_config, "PATH_TO_BE_CONFIGURED/label_map.txt", f"{obj_det_path}/data/mscoco_label_map.pbtxt")

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.compat.v2.train.Checkpoint(
        model=detection_model)
    ckpt.restore(os.path.join(model_dir, 'ckpt-0')).expect_partial()

    def get_model_detection_function(model):
        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            detections = model.postprocess(prediction_dict, shapes)

            return detections, prediction_dict, tf.reshape(shapes, [-1])

        return detect_fn

    detect_fn = get_model_detection_function(detection_model)

    category_index, label_map_dict = utils.load_labelmap(configs)

    # Get image paths
    images_paths = []
    for e in ext: images_paths += glob.glob(f'{input_dir}/*.{e}')

    # Run inference on each image
    for image_path in images_paths:

        image_np = utils.load_image_into_numpy_array(image_path)


        input_tensor = tf.convert_to_tensor(
            np.expand_dims(image_np, 0), dtype=tf.float32)
        detections, predictions_dict, shapes = detect_fn(input_tensor)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        # Use keypoints if available in detections
        keypoints, keypoint_scores = None, None
        if 'detection_keypoints' in detections:
            keypoints = detections['detection_keypoints'][0].numpy()
            keypoint_scores = detections['detection_keypoint_scores'][0].numpy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'][0].numpy(),
            (detections['detection_classes'][0].numpy() + label_id_offset).astype(int),
            detections['detection_scores'][0].numpy(),
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=200,
            min_score_thresh=.30,
            agnostic_mode=False,
            keypoints=keypoints,
            keypoint_scores=keypoint_scores,
            keypoint_edges=utils.get_keypoint_tuples(configs['eval_config']))

        cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB, image_np_with_detections)
        output_name = os.path.basename(image_path)
        cv2.imwrite(f'{output_dir}/{output_name}', image_np_with_detections)

def download_input_imags(s3, user_id:str)->str:
    # images.zipの存在確認
    if os.path.isdir('images'): shutil.rmtree('images')
    assert s3.check_uploaded_file('images.zip', f'{user_id}/input/')

    # download
    s3.download_file('images.zip', f'{user_id}/input/')
    os.path.isfile('images.zip')

    # unzip
    os.system('unzip -o images.zip')
    assert os.path.isdir('images')

    return 'images'

def upload_output(output_path:str, s3, user_id:str, output_dir:str):
    # zip output
    os.system('zip -r output.zip output')
    assert s3.exist_dir(f'{user_id}/output/')

    # upload output.zip
    if s3.check_uploaded_file('output.zip', f'{user_id}/output/'): s3.del_file('output.zip', f'{user_id}/output/')
    s3.upload_file('output.zip', f'{user_id}/output/')
    assert s3.check_uploaded_file('output.zip', f'{user_id}/output/')

def main(user_id:str = 'sample-id'):
    # download input images
    s3 = s3_utils('bounding-box-suggestion')
    input_dir:str = download_input_imags(s3, user_id)
    
    # clear output
    output_dir:str = 'output'
    if os.path.isdir(output_dir): os.system(f'rm -rf {output_dir}')
    if os.path.isfile('output.zip'): os.system(f'rm -rf output.zip')
    
    # inference
    inference(input_dir, output_dir)

    # upload output
    upload_output(output_dir, s3, user_id, output_dir)


if __name__ == '__main__':
    main()


