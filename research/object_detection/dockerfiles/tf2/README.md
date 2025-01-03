# TensorFlow Object Detection on Docker

These instructions are experimental.

## Building and running:

```bash
# From the root of the git repository
docker build -f research/object_detection/dockerfiles/tf2/Dockerfile -t od .
docker run -it od
```
