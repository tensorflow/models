# Alternative installation methods

The basic prerequisites are the same [discussed before](install). In
particular, all methods require MATLAB to be
[correctly configured with a suitable compiler](http://www.mathworks.com/help/matlab/matlab_external/changing-default-compiler.html)
(usually Xcode for Mac and GCC for Linux).

## Using the command line

If you develop MatConvNet on Mac OS X or Linus, it may be preferable
to compile the library using the command line and the supplied
`Makefile`.

### Compiling a basic version of the library

In order to compile a basic (CPU-only) version of the library use:

    > make ARCH=<your arch> MATLABROOT=<path to MATLAB>

For example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app

should work for a Mac with MATLAB R2014 installed in its default
location. The other supported architecture is `glnxa64` (for Linux).

> **Remark:** On the Mac, `vl_imreadjpeg` tool will use the native Mac OS X graphics
> library Core Graphics. On Linux, it uses instead LibJPEG.
> The tool can be disabled by defining `ENABLE_IMREADJPEG=` as an empty
> symbol. The options `IMAGELIB` option can be used to change the
> image library used (e.g. to use LibJPEG on the Mac) and the
> options `IMAGELIB_CFLAGS` and `IMAGELIB_LDFLAGS` to specify
> custom compiler and linker options.

### Using verbose and debugging modes

In order to compile in verbose mode, use the `VERB=yes` option. For example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app VERB=yes

In order to compile turning on debugging symbols and off optimizations
(useful to attach a debugger to MATLAB and debug MatConvNet), use the
`DEBUG=yes` option. For example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014a.app DEBUG=yes

### Compiling the GPU support

The default method to compile the GPU support requires a CUDA toolkit
version that matches MATLAB's internal one. Compiling may look like:

    > make ARCH=<your arch> MATLABROOT=<path to MATLAB> \
           ENABLE_GPU=yes CUDAROOT=<path to CUDA>

For example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014b.app \
           ENABLE_GPU=yes CUDAROOT=/Developer/NVIDIA/CUDA-5.5

should work on a Mac with MATLAB R2014b.

### Using an unsupported CUDA toolkit version

Use the `CUDAMETHOD=nvcc` option and the `CUDAROOT` option. For
example:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014b.app \
           ENABLE_GPU=yes CUDAROOT=/Developer/NVIDIA/CUDA-6.5 CUDAMETHOD=nvcc

Do not forget that it is now necessary to run MATLAB pointing it to
the proper CUDA toolkit libraries.

### Compiling the CuDNN support

Use the `ENABLE_CUDNN=yes` option and the `CUDNNROOT` option. From the
command line prompt, this may look like:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014b.app \
           ENABLE_GPU=yes CUDAROOT=/Developer/NVIDIA/CUDA-6.5 CUDAMETHOD=nvcc \
           ENABLE_CUDNN=yes CUDNNROOT=local/


### Further examples

Compiling all the features in MatConvNet in Mac OS X with MATLAB
R2014b, CUDA 6.5 and CuDNN Candidate Release 2:

    > make ARCH=maci64 MATLABROOT=/Applications/MATLAB_R2014b.app \
           ENABLE_GPU=yes CUDAROOT=/Developer/NVIDIA/CUDA-6.5 CUDAMETHOD=nvcc \
           ENABLE_CUDNN=yes CUDNNROOT=local/
           ENABLE_IMREADJPEG=yes \
           LIBJPEG_INCLUDE=/opt/local/include \
           LIBJPEG_LIB=/opt/local/lib
