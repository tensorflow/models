# Frequently-asked questions (FAQ)

## Running MatConvNet

### Do I need a specific version of the CUDA devkit?

Officially, MathWorks supports a specific version of the CUDA devkit
with each MATLAB version (see [here](install.md#gpu)). However, in
practice we normally use the most recent version of CUDA (and cuDNN)
available from NVIDIA without problems (see
[here](install.md#nvcc)).

### Can I use MatConvNet with CuDNN?

Yes, and this is the recommended way of running MatConvNet on NVIDIA
GPUs. However, you need to install cuDNN and link it to
MatConvNet. See the [installation instructions](install.md#cudnn) to
know how.

### How do I fix the error `Attempt to execute SCRIPT vl_nnconv as a function`?

Before the toolbox can be used, the
[MEX files](http://www.mathworks.com/support/tech-notes/1600/1605.html
) must be compiled. Make sure to follow the
[installation instructions](install.md). If you have done so and the
MEX files are still not recognized, check that the directory
`matlab/toolbox/mex` contains the missing files. If the files are
there, there may be a problem with the way MEX files have been
compiled.

### Why files such as `vl_nnconv.m` do not contain any code?

Functions such as `vl_nnconv`, `vl_nnpool`, `vl_nnbnorm` and many
others are implemented MEX files. In this case, M files such as
`vl_nnconv.m` contain only the function documentation. The code of the
function is actually found in `matlab/src/vl_nnconv.cu` (a CUDA/C++
source file) or similar.

### Why do I get compilation error `error: unrecognized command line option "-std=c++11"` on a Linux machine?

This is caused by an incompatible version of GCC compiler
([<4.6](https://gcc.gnu.org/projects/cxx-status.html#cxx11)) with your MATLAB.
You can either install a newer version of GCC (if available), or you
can force MATLAB not to use the offending compiler option and replace it with
the previous name of the C++11 standard argument:
 * In MATLAB run: `mex -setup c++`.
 * Run `edit(fullfile(prefdir, 'mex_C++_glnxa64.xml'))` to edit your MATLAB
 compiler options.
 * Replace all occurrences of `-std=c++11` with `-std=c++0x` and save the file.
