import cv2
import numpy as np
import tensorflow as tf

IMAGE_PATH = '/home/opencv-mds/OpenCV_in_Ubuntu/Data/Lane_Detection_Images/test.png'
IMAGE_PATH_2 = '/home/opencv-mds/testOpenCV/OpenCV_in_Ubuntu/Data/TrafficLight_Detection/green_light_02.png'
PB_PATH = '/home/opencv-mds/models/frozen_inference_graph.pb'
VIDEO_PATH = '/home/opencv-mds/OpenCV_in_Ubuntu/Data/Lane_Detection_Videos/challenge.mp4'
VIDEO_PATH_2 = '/home/opencv-mds/Project_Videos/ETS_H_01.mp4'




def frameProcessing(image, graph):
    return image
    

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


Video(VIDEO_PATH, ##PB_GRAPH##)
   







