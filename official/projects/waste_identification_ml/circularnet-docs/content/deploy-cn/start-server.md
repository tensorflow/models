Follow these steps to create a Triton inference server and load the pixel-level
instance segmentation models from the repository to the server:

**Note:** If you are in Google Cloud but have closed your **SSH-in-browser**
window, open the **VM instances** page and click **SSH** in the row of the
NVIDIA T4 GPU instance you want to connect to. Then, let the **SSH-in-browser**
tool open. For more information, see [Connect to VMs](https://cloud.google.com/compute/docs/connect/standard-ssh#connect_to_vms).

1. On the **SSH-in-browser** window of your VM instance in Google Cloud or the terminal of your edge device, open the `server` folder in the `prediction_pipeline` directory:

    ```
    cd models/official/projects/waste_identification_ml/docker_solution/prediction_pipeline/server/
    ```

1. Run the `triton_server.sh` script to create the Triton inference server and load the models to the server:

    ```
    bash triton_server.sh
    ```

    This script loads as many models as you want at the same time. Later, you can choose which model you want to send your request to from the client side. For more information, see [Prepare and analyze images](/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/).

    For example, when you start analyzing images, you can send them from the
    client to the following models in the Triton server you created:

    -  `material_resnet_v2_512_1024`: shows the material and its subtype using
       ResNet for classification on images of 512 x 1024 pixels.
    -  `material_form_resnet_v2_512_1024`: shows the form of an object, for
       example, can or bottle, using ResNet for classification on images of 512
       x 1024 pixels.
    -  `material_mobilenet_v2_512_1024`: shows the material and its subtype
       using MobileNet for classification on images of 512 x 1024 pixels.
    -  `material_form_mobilenet_v2_512_1024`: shows the form of an object, for
       example, can or bottle, using MobileNet for classification on images of
       512 x 1024 pixels.

You have finished setting up the Triton inference server. The server keeps
running on the backend and your terminal window lets you run new commands to
interact with it.

You can confirm the server is running by opening a screen session:

1. List the `screen` sessions:

    ```
    screen -ls
    ```

      The output from your server shows the `(Detached)` message because you are
      outside of the session.

1. Enter the `screen` session for the server:

    ```
    screen -r server
    ```

    The `screen` session opens and displays the ongoing operations on the
    server. The models must show a `READY` status on the `screen` session when
    they are successfully deployed.

1. If you want to exit the screen session without stopping the server, press
   **Ctrl + A + D** keys.

## What's next

-  [Prepare and analyze images](/official/projects/waste_identification_ml/circularnet-docs/content/analyze-data/)