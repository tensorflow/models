Follow these steps to start a Triton inference server and configure it to serve
inference requests to the circularnet model.

**Note:** If you are in Google Cloud but have closed your **SSH-in-browser**
window, open the **VM instances** page and click **SSH** in the row of the
NVIDIA T4 GPU instance you want to connect to. Then, let the **SSH-in-browser**
tool open. For more information, see [Connect to VMs](https://cloud.google.com/compute/docs/connect/standard-ssh#connect_to_vms).

1. On the **SSH-in-browser** window of your VM instance in Google Cloud or the
terminal of your edge device, open the `server` folder in the
`waste_identification_ml` project directory:

    ```
    cd models/official/projects/waste_identification_ml/Triton_TF_Cloud_Deployment/server/
    ```

1. Run the `triton_server.sh` script to create the Triton inference server and
load the most recent circularnet model on the server:

    ```
    bash triton_inference_server.sh
    ```

The server keeps running in the background (using a screen session).

You can confirm the server is running by opening the screen session the server
is running within:

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
    server. The model will show a `READY` status when it is successfully
    deployed.

1. If you want to exit the screen session without stopping the server, press
   **Ctrl + a** then **d** keyboard shortcut. This will detach the session.

## What's next

Next, you are ready to send your images for inference. See the
[start client](start-client.md) section.