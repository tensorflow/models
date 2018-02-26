## Overview

This sample is an implementation of the "fast gradient sign method" in TensforFlow for generating adversarial images.  For details see [blogpost]().

## Steps

1. Install requirements.

    ```
    pip install requirements.txt
    ```

1. Download a trained MobileNet model.

    ```
    python get_model.py
    ```

1. Generate an adversarial image.

    ```
    python adversarial_image.py [ORIGINAL_IMAGE_FILENAME]
    ```

1. When the process is finished, 3 output images files will be created:

    - `original.png` is the original image file saved in the PNG format.

    - `out.png` is the perturbed image.

    - `signg.png` is the sign of gradient.

## Optional parameters

- To perturb the input image 10 times:

    ```
    python adversarial_image.py [IMAGE_FILENAME] --n-iter 10
    ```

    Only the final `output.png` and the last sign of gradient will be saved.


- To adjust the portion of the sign of gradient to be applied:

    ```
    python adversarial_image.py [IMAGE_FILENAME] --epsilon 0.001
    ```

- These two optional parameters can be used together:

    ```
    python adversarial_image.py [IMAGE_FILENAME] --n-iter 10 --epsilon 0.001
    ```
