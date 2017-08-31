# Developer notes

## Understand MatConvNet and contribute to it

There are several parts to MatConvNet. It is important to understand
this modular structure so that everybody can maximally benefit from
your contributions. In particular, the easiest way to contribute
something of general utility to the library is to create a new
building block, as explained below:

1.  **Building blocks.** The most important are the neural network
    building blocks, i.e. MATLAB functions prefixed by `vl_nn` such as
    `vl_nnconv`. To add a new "layer" or "computational block", add
    such a function. These are found in the directory `matlab/`.

2.  **CNN wrappers.** There are a few wrappers (`vl_simplenn` and
    `vl_dagnn`) that allow implementing networks composed of several
    blocks. These are located in `matlab/wrappers`. These wrappers are
    fairly generic, but there are case in which writing a new wrapper
    type may be useful.

3.  **C++/CUDA library.** While many blocks can be written in MATLAB
    directly, some such as convolution and max-pooling require custom
    C++ and CUDA code. In this case, create a MEX file in `matlab/src`
    and add the required functionality to the C++/CUDA library by
    adding files to `matlab/src/bits`.

4.  **Examples.** Other components of the library are at the moment
    delivered as examples in the `examples/` directory. This include
    basic SGD code for training networks in
    `examples/cnn_train.m`. Note that these functions are *not*
    supposed to have very broad generality, although in practice they
    can be an excellent starting point to write your own learning
    algorithms.

If you develop the C++/CUDA core of the library, it may be convenient
to switch using an IDE/command line and the
[Makefile for compiling the library](install-alt.md).

## Creating the website

MatConvNet website uses the
[MkDocs documentation system](http://www.mkdocs.org), a simple static
website generator written in Python. In order to setup the necessary
software, follow these steps:

1.  Install MkDocs as explained
    [here](http://www.mkdocs.org/#installation).
2.  Install the
    [MathJax MkDocs extension](https://github.com/mayoff/python-markdown-mathjax). This
    is explained in the `README` file in the extension GitHub
    repository and consists in renaming and copying a file into your
    copy of MkDocs.
3.  Make sure that the `mkdocs` command is executable from the command line, or otherwise
    modify the file `doc/Makefile` in MatConvNet to point to the correct executable.

At this point, the website can be built with `make doc-site` from the
command line. The documentation is created in `doc/site/site` from the
`doc/site/docs/*.md` files.

For developing the website, it is useful to start the MkDocs HTTP server as follows:

    cd doc/site
    mkdocs serve

At this point visit `http://127.0.0.1:8000/` to see a copy of the
current website.

### Creating the MATLAB function pages

The documentation for the MATLAB function is extracted automatically
from the m-files using the `doc/matdoc.py` Python script. This is
called by `doc/Makefile` as needed.

`matdoc.py` does three things: it extracts the function documentation
from a m-file, it interprets it using a pseudo-Markdown syntax, and
then it writes the result in Markdown format for use with
MkDocs. These are stored in `doc/site/docs/mfiles`.
