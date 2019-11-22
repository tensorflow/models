# correlation_cost

FEELVOS uses correlation_cost as an optional dependency to improve the speed and memory consumption
of cross-correlation.

## Installation

Unfortunately we cannot provide the code for correlation_cost directly, so you
will have to copy some files from this pull request
https://github.com/tensorflow/tensorflow/pull/21392/. For your convenience we
prepared scripts to download and adjust the code automatically.

In the best case, all you need to do is run compile.sh with the path to your
CUDA installation (tested only with CUDA 9).
Note that the path should be to a folder containing the cuda folder, not to the
cuda folder itself, e.g. if your cuda is in /usr/local/cuda-9.0, you can create
a symlink /usr/local/cuda pointing to /usr/local/cuda-9.0 and then run

```bash
sh build.sh /usr/local/
```

This will

* Download the code via ```sh get_code.sh ```
* Apply minor adjustments to the code via ```sh fix_code.sh```
* Clone the dependencies cub and thrust from github via ```sh clone_dependencies.sh```
* Compile a shared library correlation_cost.so for correlation_cost via
```sh compile.sh "${CUDA_DIR}"```

Please review the licenses of correlation_cost, cub, and thrust.

## Enabling correlation_cost
If you managed to create the correlation_cost.so file, then set
```USE_CORRELATION_COST = True``` in feelvos/utils/embedding_utils.py and try to run
```sh eval.sh```.
