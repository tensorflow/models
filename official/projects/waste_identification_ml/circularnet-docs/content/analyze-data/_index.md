CircularNet provides two image-analysis models. The first detects _material
types_, and the second detects _material forms_. These models utilize a Mask
R-CNN algorithm for image training and implement ResNet or MobileNet as the
convolutional neural networks for image classification tasks.

The models are loaded sequentially to achieve accurate predictions. When working
with images, each image undergoes preprocessing before the models use them for
prediction. In the case of video files, the video is split into individual
frames at a given frame rate. These frames are then processed in the same
sequential manner as images.

The predictions from the two models result in two distinct outputs, which are
then post-processed and combined into a single comprehensive output. This output
includes critical information such as the number of detected objects, their
bounding boxes, class names, class IDs, and masks for each object. Further
computer vision techniques extract multiple properties of each object, including
color detection. These properties facilitate object tracking and help eliminate
duplicate object counts.

If you [deploy the server](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/start-server) on Google Cloud, you can
automate the entire image analysis workflow within your VM instance. Integration
with BigQuery tables, storage buckets, and dashboards allows for seamless data
flow and real-time updates. A [prediction pipeline](./learn-about-pipeline) for
Google Cloud pushes the data directly to storage buckets and BigQuery tables,
which you can connect to the dashboard for [visualization and analysis](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/).

On the other hand, direct data transfer to the cloud for edge device implementations needs a client-side configuration. A [prediction pipeline](/official/projects/waste_identification_ml/circularnet-docs/content/learn-about-pipeline) for devices lets you load models sequentially and store image analysis results locally.

This section describes how to apply the two specialized CircularNet models using
a prediction pipeline on the client side to prepare and analyze the images you
capture.