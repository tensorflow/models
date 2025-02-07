After meeting the [prerequisites](/official/projects/waste_identification_ml/circularnet-docs/content/before-you-begin), follow these steps to clone the project from the [GitHub repository](https://github.com/tensorflow/models/tree/master/official/projects/waste_identification_ml) and install all the required packages.

Run the following commands on the **SSH-in-browser** window of your VM instance
in Google Cloud or the terminal of your edge device:

1. Install Git:

    ```
    sudo apt-get install git
    ```

    **Note:** If the `Do you want to continue?` message is displayed, enter `Y` and press **Enter** to continue.<br><br>

1. Clone the [GitHub repository](https://github.com/tensorflow/models/tree/master/official/projects/waste_identification_ml):

    ```
    git clone --depth 1 https://github.com/tensorflow/models.git
    ```

    **Note:** If the scripts from this guide are unavailable on GitHub, the
    CircularNet team provides zip files for you to download locally. In that
    case, cloning from GitHub is unnecessary; you only have to unzip the
    files.<br><br>

1. Open the `client` folder in the `prediction_pipeline` directory:

    ```
    cd models/official/projects/waste_identification_ml/docker_solution/prediction_pipeline/client/
    ```

1. Run the `requirements.sh` script to install all the required packages and libraries:

    ```
    sh requirements.sh
    ```

    **Note:** If the `Do you want to continue?` message is displayed, enter `Y` and press **Enter** to continue.<br><br>

1. Return to the root directory:

    ```
    cd\
    ```

    Then, press **Enter**.