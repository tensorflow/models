import cv2
import numpy as np
import tensorflow as tf

IMAGE_PATH = '/home/mds/OpenCV_in_Ubuntu/Data/Lane_Detection_Images/test.png'
PB_PATH = '/home/mds/models/frozen_inference_graph.pb'

def import_graph(PATH_TO_FTOZEN_PB):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_FTOZEN_PB, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

def run_inference_for_single_image(image, sess, image_tensor, tensor_dict):
    # Run inference
    output_dict = sess.run(tensor_dict,feed_dict={image_tensor: image})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict

def open_image_and_inference(image_path, detection_graph):
    with detection_graph.as_default():
        with tf.Session() as sess:
        # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
                ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            image = cv2.imread(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            output_dict = run_inference_for_single_image(image_np_expanded, sess, image_tensor, tensor_dict)
            num_detections = int(output_dict['num_detections'])
            rows = image_np.shape[0]
            cols = image_np.shape[1]
            for i in range(num_detections):
                classId = int(output_dict['detection_classes'][i])
                score = float(output_dict['detection_scores'][i])
                bbox = [float(v) for v in output_dict['detection_boxes'][i]]
                if score > 0.4:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    cv2.rectangle(image, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            cv2.namedWindow("result", cv2.WINDOW_NORMAL)
            cv2.imshow("result", image)
            cv2.waitKey()
    
open_image_and_inference(IMAGE_PATH, import_graph(PB_PATH))