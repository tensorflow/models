# Contributing guidelines

## How to contribute to MatConvNet

For a description of how the library is structured, take a look at the
[Developers notes](http://www.vlfeat.org/matconvnet/developers/) on
the MatConvNet website.

### Issues

We are grateful for any reported issues which help to remove bugs and
improve the overall quality of the library. In particular, you can use
the issue tracker to:

* report bugs and unexpected crashes
* discuss library design decisions
* request new features

When reporting bugs, it really helps if you can provide the following:

* Which steps are needed to reproduce the issue
* MATLAB, compiler and CUDA version (where appropriate)

Before opening an issue to report a bug, please make sure that the bug
is reproducible on the latest version of the master branch.

The most difficult bugs to remove are those which cause crashes of the
core functions (e.g. CUDA errors etc.). In those cases, it is really
useful to create a *minimal example* which is able to reproduce the
issue. We know that this may mean a bit of work, but it helps us to
remove the bug more quickly.

### Pull requests

Please make any Pull Requests against the `devel` branch rather than
the `master` branch which is maintained as the latest stable release
of the library.

As a general rule, it is much easier to accept small Pull Requests
that make a single improvement to the library than complex code
changes that affect multiple parts of the library. When submitting
substantial changes, it is useful if unit tests are provided with the
code.
