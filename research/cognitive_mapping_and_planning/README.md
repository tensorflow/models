# Cognitive Mapping and Planning for Visual Navigation
**Saurabh Gupta, James Davidson, Sergey Levine, Rahul Sukthankar, Jitendra Malik**

**Computer Vision and Pattern Recognition (CVPR) 2017.**

**[ArXiv](https://arxiv.org/abs/1702.03920), 
[Project Website](https://sites.google.com/corp/view/cognitive-mapping-and-planning/)**

### Citing
If you find this code base and models useful in your research, please consider
citing the following paper:
  ```
  @inproceedings{gupta2017cognitive,
    title={Cognitive Mapping and Planning for Visual Navigation},
    author={Gupta, Saurabh and Davidson, James and Levine, Sergey and
      Sukthankar, Rahul and Malik, Jitendra},
    booktitle={CVPR},
    year={2017}
  }
  ```

### Contents
1.  [Requirements: software](#requirements-software)
2.  [Requirements: data](#requirements-data)
3.  [Test Pre-trained Models](#test-pre-trained-models)
4.  [Train your Own Models](#train-your-own-models)

### Requirements: software
1.  Python Virtual Env Setup: All code is implemented in Python but depends on a
    small number of python packages and a couple of C libraries. We recommend
    using virtual environment for installing these python packages and python
    bindings for these C libraries.
      ```Shell
      VENV_DIR=venv
      pip install virtualenv
      virtualenv $VENV_DIR
      source $VENV_DIR/bin/activate
      
      # You may need to upgrade pip for installing openv-python.
      pip install --upgrade pip
      # Install simple dependencies.
      pip install -r requirements.txt

      # Patch bugs in dependencies.
      sh patches/apply_patches.sh
      ```

2.  Install [Tensorflow](https://www.tensorflow.org/) inside this virtual
    environment. You will need to use one of the latest nightly builds 
    (see instructions [here](https://github.com/tensorflow/tensorflow#installation)).

3.  Swiftshader: We use
    [Swiftshader](https://github.com/google/swiftshader.git), a CPU based
    renderer to render the meshes.  It is possible to use other renderers,
    replace `SwiftshaderRenderer` in `render/swiftshader_renderer.py` with
    bindings to your renderer. 
    ```Shell
    mkdir -p deps
    git clone --recursive https://github.com/google/swiftshader.git deps/swiftshader-src
    cd deps/swiftshader-src && git checkout 91da6b00584afd7dcaed66da88e2b617429b3950
    mkdir build && cd build && cmake .. && make -j 16 libEGL libGLESv2
    cd ../../../
    cp deps/swiftshader-src/build/libEGL* libEGL.so.1
    cp deps/swiftshader-src/build/libGLESv2* libGLESv2.so.2
    ```

4.  PyAssimp: We use [PyAssimp](https://github.com/assimp/assimp.git) to load
    meshes.  It is possible to use other libraries to load meshes, replace
    `Shape` `render/swiftshader_renderer.py` with bindings to your library for
    loading meshes. 
    ```Shell
    mkdir -p deps
    git clone https://github.com/assimp/assimp.git deps/assimp-src
    cd deps/assimp-src
    git checkout 2afeddd5cb63d14bc77b53740b38a54a97d94ee8
    cmake CMakeLists.txt -G 'Unix Makefiles' && make -j 16
    cd port/PyAssimp && python setup.py install
    cd ../../../..
    cp deps/assimp-src/lib/libassimp* .
    ```

5.  graph-tool: We use [graph-tool](https://git.skewed.de/count0/graph-tool)
    library for graph processing.
    ```Shell
    mkdir -p deps
    # If the following git clone command fails, you can also download the source
    # from https://downloads.skewed.de/graph-tool/graph-tool-2.2.44.tar.bz2
    git clone https://git.skewed.de/count0/graph-tool deps/graph-tool-src
    cd deps/graph-tool-src && git checkout 178add3a571feb6666f4f119027705d95d2951ab
    bash autogen.sh
    ./configure --disable-cairo --disable-sparsehash --prefix=$HOME/.local
    make -j 16
    make install
    cd ../../
    ```

### Requirements: data
1.  Download the Stanford 3D Indoor Spaces Dataset (S3DIS Dataset) and ImageNet
    Pre-trained models for initializing different models. Follow instructions in
    `data/README.md`

### Test Pre-trained Models
1.  Download pre-trained models. See `output/README.md`.

2.  Test models using `scripts/script_test_pretrained_models.sh`.

### Train Your Own Models
All models were trained asynchronously with 16 workers each worker using data
from a single floor. The default hyper-parameters correspond to this setting.
See [distributed training with
Tensorflow](https://www.tensorflow.org/deploy/distributed) for setting up
distributed training. Training with a single worker is possible with the current
code base but will require some minor changes to allow each worker to load all
training environments.

### Contact
For questions or issues open an issue on the tensorflow/models [issues
tracker](https://github.com/tensorflow/models/issues). Please assign issues to
@s-gupta.

### Credits
This code was written by Saurabh Gupta (@s-gupta).
