import cv2
import numpy as np
import tensorflow as tf

IMAGE_PATH = '/home/opencv-mds/OpenCV_in_Ubuntu/Data/Lane_Detection_Images/test.png'
IMAGE_PATH_2 = '/home/opencv-mds/testOpenCV/OpenCV_in_Ubuntu/Data/TrafficLight_Detection/green_light_02.png'
PB_PATH = '/home/opencv-mds/models/frozen_inference_graph.pb'
VIDEO_PATH = '/home/opencv-mds/OpenCV_in_Ubuntu/Data/Lane_Detection_Videos/challenge.mp4'
VIDEO_PATH_2 = '/home/opencv-mds/Project_Videos/ETS_H_01.mp4'

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
          'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            # Run inference
            output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: image})

            # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
    return output_dict
  
    
def draw_inference_result(image, output_dict):
    result = np.copy(image)
    num_detections = int(output_dict['num_detections'])
    rows = result.shape[0]
    cols = result.shape[1]
    result = np.copy(image)
    for i in range(num_detections):
        classId = int(output_dict['detection_classes'][i])
        score = float(output_dict['detection_scores'][i])
        bbox = [float(v) for v in output_dict['detection_boxes'][i]]
        if score > 0.1:
            x = bbox[1] * cols
            y = bbox[0] * rows
            right = bbox[3] * cols
            bottom = bbox[2] * rows
            cv2.rectangle(result, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
    return result


def open_pb(PATH):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph


def frameProcessing(image, graph):
    image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_np_expanded = np.expand_dims(image_np, axis=0)
    output_dict = run_inference_for_single_image(image_np_expanded, graph)
    result = draw_inference_result(image, output_dict)
    return result
    

def Video(openpath, graph, savepath = "output.avi"):
    cap = cv2.VideoCapture(openpath)
    if cap.isOpened():
        print("Video Opened")
    else:
        print("Video Not Opened")
        print("Program Abort")
        exit()
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    out = cv2.VideoWriter(savepath, fourcc, fps, (width, height), True)
    cv2.namedWindow("Input", cv2.WINDOW_GUI_EXPANDED)
    cv2.namedWindow("Output", cv2.WINDOW_GUI_EXPANDED)
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Our operations on the frame come here
            output = frameProcessing(frame, graph)
            # Write frame-by-frame
            out.write(output)
            # Display the resulting frame
            cv2.imshow("Input", frame)
            cv2.imshow("Output", output)
        else:
            break
        # waitKey(int(1000.0/fps)) for matching fps of video
        if cv2.waitKey(int(1000.0/fps)) & 0xFF == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    return


Video(VIDEO_PATH, open_pb(PB_PATH))
   







