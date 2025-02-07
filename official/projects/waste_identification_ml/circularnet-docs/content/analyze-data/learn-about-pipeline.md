CircularNet offers prediction pipelines for processing, analyzing, and
performing object recognition on video or image files. These pipelines
facilitate systematic and automated video and image analysis using the Mask
R-CNN algorithm and additional object detection and feature extraction
processes.

You can run a prediction pipeline from a script to automatically apply the two
specialized models that analyze images or video frames. A prediction pipeline
operates through the following series of actions in a specific order to ensure
reliable and consistent results:

1. Organize your videos or images chronologically according to their creation
   time or another time-related metadata.
1. Import files one at a time. If your files are videos, the pipeline decomposes
   them into individual frames and runs two Mask R-CNN models for pixel-level
   instance segmentation on each frame.
1. Implement a color detection algorithm to identify and categorize the colors
   of the detected objects within the frames or images.
1. Extract and record features from the detected objects, facilitating analysis
   and machine learning applications.
1. Output prediction results of each frame or image with overlaid masks and
   identification of detected objects.

After processing all frames of a single video, the pipeline implements an
object-tracking algorithm to identify and eliminate duplicate occurrences of
objects across sequential frames, enhancing the accuracy of object detection and
analysis.

Moreover, [applying a prediction pipeline in Google Cloud](/official/projects/waste_identification_ml/circularnet-docs/content/prediction-pipeline-in-cloud) automatically uploads raw images and prediction results into [BigQuery](https://cloud.google.com/bigquery) tables. This seamless integration allows you to combine [visualization dashboards with analytical reports](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/) effortlessly.