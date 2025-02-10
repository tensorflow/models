## Table of contents

**[CircularNet overview](#circularnet-overview)**

  * [Get started with CircularNet](#get-started-with-circularnet)

**[Discover CircularNet](/official/projects/waste_identification_ml/circularnet-docs/content/discover-cn/_index.md)**

* [Benefits of CircularNet](/official/projects/waste_identification_ml/circularnet-docs/content/discover-cn/benefits-of-cn.md)
* [When to use CircularNet](/official/projects/waste_identification_ml/circularnet-docs/content/discover-cn/when-to-use-cn.md)
* [How CircularNet works](/official/projects/waste_identification_ml/circularnet-docs/content/discover-cn/how-cn-works.md)

**[Choose a deployment solution](/official/projects/waste_identification_ml/circularnet-docs/content/solutions/_index.md)**

* [Where to host the models](/official/projects/waste_identification_ml/circularnet-docs/content/solutions/_index.md#where-to-host-the-models)
    * [Cloud deployment](/official/projects/waste_identification_ml/circularnet-docs/content/solutions/_index.md#cloud-deployment)
    * [Edge device deployment](/official/projects/waste_identification_ml/circularnet-docs/content/solutions/_index.md#edge-device-deployment)
    * [In-house server deployment](/official/projects/waste_identification_ml/circularnet-docs/content/solutions/_index.md#in-house-server-deployment)

**[Set up the system requirements](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/_index.md)**

* [Choose a camera](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/_index.md)
    * [Recommendations for selecting a machine vision camera](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/camera-recommendations.md)
    * [Factors based on vision system placement](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/factors.md)
        * [Sensor size](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/factors.md#sensor-size)
        * [Focal length](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/factors.md#focal-length)
        * [Aperture size (f-number)](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/factors.md#aperture-size-f-number)
        * [Shutter speed](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/factors.md#shutter-speed)
    * [Table of specifications](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/table-of-specs.md)
* [Choose edge device hardware](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-edge-device/_index.md)

**[Deploy CircularNet](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/_index.md)**

* [Before you begin](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/before-you-begin.md)
* [Clone the repository and install packages](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/clone-repo.md)
* [Start the server](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/start-server.md)

**[Prepare and analyze images](/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/_index.md)**

* [Learn about the prediction pipeline](/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/learn-about-pipeline.md)
* [Apply the prediction pipeline in Google Cloud](/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/prediction-pipeline-in-cloud.md)
* [Apply the prediction pipeline in an edge device](/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/prediction-pipeline-in-edge.md)

**[View data analysis and reporting](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/_index.md)**

* [Before you begin](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/before-you-begin.md)
* [Configure the dashboard](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/configure-dashboard.md)

## CircularNet overview

CircularNet is a free computer vision model developed by Google that utilizes
artificial intelligence (AI) and machine learning (ML) to provide detailed and
accurate identification of waste streams and recyclables. Trained on a diverse
global dataset, CircularNet aims to make waste management analytics accessible
and promote data-driven decision-making. It supports efforts to keep valuable
resources out of landfills and in circulation. Open access and collaboration are
fundamental to CircularNet's vision. Its open-source models, powered by
[TensorFlow](https://www.tensorflow.org/) and available on
[GitHub](https://github.com/tensorflow/models/tree/master/official/projects/waste_identification_ml),
are free to use, customizable, and can help bring analytics to new markets while
minimizing cost.

This guide offers step-by-step instructions for setting up and integrating
CircularNet, accommodating various deployment options. It describes different
deployment preferences so you can install CircularNet according to your needs.

### Get started with CircularNet

Start exploring CircularNet by reviewing the following documentation:

1. Discover [the benefits, features, components, and use cases](/official/projects/waste_identification_ml/circularnet-docs/content/discover-cn/) of CircularNet.
1. Choose between [the different deployment options](/official/projects/waste_identification_ml/circularnet-docs/content/solutions/) to install CircularNet models.
1. Learn about [the recommendations for installing the camera](/official/projects/waste_identification_ml/circularnet-docs/content/system-req/choose-camera/) you require to capture images.
1. Follow a step-by-step solution example to [deploy CircularNet](/official/projects/waste_identification_ml/circularnet-docs/content/deploy-cn/) and [prepare your captured images](/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/) for analysis and object tracking.
1. Learn how to connect your data with [a dashboard for visualization and reporting](/official/projects/waste_identification_ml/circularnet-docs/content/view-data/).