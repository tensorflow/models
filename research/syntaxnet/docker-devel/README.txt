Docker is used for packaging the SyntaxNet. There are three primary things we
build with Docker,

1. A development image, which contains all source built with Bazel.
2. Python/pip wheels, built by running a command in the development container.
3. A minified image, which only has the compiled version of TensorFlow and
   SyntaxNet, by installing the wheel built by the above step.


Important info (please read)
------------------------------

One thing to be wary of is that YOU CAN LOSE DATA IF YOU DEVELOP IN A DOCKER
CONTAINER. Please be very careful to mount data you care about to Docker
volumes, or use a volume mount so that it's mapped to your host filesystem.

Another note, especially relevant to training models, is that Docker sends the
whole source tree to the Docker daemon every time you try to build an image.
This can take some time if you have large temporary model files lying around.
You can exclude your model files by editing .dockerignore, or just don't store
them in the base directory.


Step 1: Building the development image
------------------------------

Simply run `docker build -t dragnn-oss .` in the base directory. Make sure you
have all the source checked out correctly, including git submodules.


Step 2: Building wheels
------------------------------

Please run,

  bash ./docker-devel/build_wheels.sh

This actually builds the image from Step 1 as well.


Step 3: Building the development image
------------------------------

First, ensure you have the file

  syntaxnet_with_tensorflow-0.2-cp27-cp27mu-linux_x86_64.whl

in your working directory, from step 2. Then run,

  docker build -t dragnn-oss:latest-minimal -f docker-devel/Dockerfile.min .

If the filename changes (e.g. you are on a different architecture), just update
Dockerfile.min.


Developing in Docker
------------------------------

We recommend developing in Docker by using the `./docker-devel/build_devel.sh`
script; it will set up a few volume mounts, and port mappings automatically.
You may want to add more port mappings on your own. If you want to drop into a
shell instead of launching the notebook, simply run,

  ./docker-devel/build_devel.sh /bin/bash
