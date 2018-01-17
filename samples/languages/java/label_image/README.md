# Image Classification Example

1. Download the model:
   - If you have [TensorFlow 1.4+ for Python installed](https://www.tensorflow.org/install/),
     run `python ./download.py`
   - If not, but you have [docker](https://www.docker.com/get-docker) installed,
     run `download.sh`.

2. Compile [`LabelImage.java`](src/main/java/LabelImage.java):

   ```
   mvn compile
   ```

3. Download some sample images:
   If you already have some images, great. Otherwise `download_sample_images.sh`
   gets a few.

3. Classify!

   ```
   mvn -q exec:java -Dexec.args="<path to image file>"
   ```
