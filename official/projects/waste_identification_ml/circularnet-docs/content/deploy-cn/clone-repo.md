After meeting the [prerequisites](before-you-begin.md),
follow these steps to clone the project from the [GitHub repository](https://github.com/tensorflow/models/tree/master/official/projects/waste_identification_ml) and install all the required packages.

Run the following commands on the **SSH-in-browser** window of your VM instance
in Google Cloud or the terminal of your edge device:

1. Install Git:

    ```
    sudo apt-get install git
    ```

1. Clone the tensorflow models, which contains Circularnet
(waste_identification_ml) [Repo link](https://github.com/tensorflow/models)

    ```
    git clone --depth 1 https://github.com/tensorflow/models.git
    ```

1. Open the `client` folder within the `waste_identification_ml` project
directory:

    ```
    cd models/official/projects/waste_identification_ml/Triton_TF_Cloud_Deployment/client/
    ```

1. Run `requirements.sh` to install all the required packages and libraries:

    ```
    sh requirements.sh
    ```

1. Return to the root directory:

    ```
    cd\
    ```

Next, start the triton inference server that will serve inference requests. [Start server](start-server.md)