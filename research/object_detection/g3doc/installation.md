# Installation

It is recommended to use the [Anaconda Python distribution](https://www.anaconda.com/downloa) to install and use the `object_detection` API. Then you can setup a new Conda environment with all the necessary dependencies:

``` bash
wget https://raw.githubusercontent.com/hadim/models/master/research/object_detection/environment.yml
conda env create -f environment.yml

source activate object_detection
pip install object_detection

# or to install from a local directory
# pip install -e .
```

Check the installation is correct by running tests:

```bash
model_builder_test.py
```
