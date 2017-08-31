#
# mexopts.sh	Shell script for configuring MEX-file creation script,
#               mex, to use NVCC for building GPU MEX files.
#
# usage:        Do not call this file directly; it is sourced by the
#               mex shell script.  Modify only if you don't like the
#               defaults after running mex.  No spaces are allowed
#               around the '=' in the variable assignment.
#
# Note: For the version of system compiler supported with this release,
#       refer to the Supported and Compatible Compiler List at:
#       http://www.mathworks.com/support/compilers/current_release/
#
#
# SELECTION_TAGs occur in template option files and are used by MATLAB
# tools, such as mex and mbuild, to determine the purpose of the contents
# of an option file. These tags are only interpreted when preceded by '#'
# and followed by ':'.
#
#SELECTION_TAG_MEX_OPT: Template Options file for building nvcc MEX-files
#
# Copyright 2012 The MathWorks, Inc.
#----------------------------------------------------------------------------
#
    TMW_ROOT="$MATLAB"
    MFLAGS=''
    # Set the value of $NVCC. By default we assume that it's on your path. You
    # can explicitly provide the path in an environment variable if that's not
    # the case.
    if [ -n "$MW_NVCC_PATH" ]; then
        NVCC="$MW_NVCC_PATH"
    else
        NVCC="nvcc"
    fi
    if [ "$ENTRYPOINT" = "mexLibrary" ]; then
        MLIBS="-L$TMW_ROOT/bin/$Arch -lmx -lmex -lmat -lmwservices -lut"
    else
        MLIBS="-L$TMW_ROOT/bin/$Arch -lmx -lmex -lmat"
    fi
    case "$Arch" in
        Undetermined)
#----------------------------------------------------------------------------
# Change this line if you need to specify the location of the MATLAB
# root directory.  The script needs to know where to find utility
# routines so that it can determine the architecture; therefore, this
# assignment needs to be done while the architecture is still
# undetermined.
#----------------------------------------------------------------------------
            MATLAB="$MATLAB"
            ;;
        glnxa64)
#----------------------------------------------------------------------------
            RPATH="-Wl,-rpath-link,$TMW_ROOT/bin/$Arch"
            # StorageVersion: 1.0
            # CkeyName: nvcc
            # CkeyManufacturer: NVIDIA
            # CkeyLanguage: C CUDA
            # CkeyVersion:
            # CkeyLinkerName: GNU ld
            # CkeyLinkerVersion:
            # $NVCC was set above.
            CC='$NVCC'
            CFLAGS=" -ccbin=gcc-4.4 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=\\\"sm_30,compute_30\\\" -I$TMW_ROOT/toolbox/distcomp/gpu/extern/include --compiler-options=-ansi,-D_GNU_SOURCE,-fexceptions,-fPIC,-fno-omit-frame-pointer,-pthread"
            CLIBS="$RPATH $MLIBS -lm -lmwgpu $TMW_ROOT/bin/$Arch/libcudart.so.5.0"
            COPTIMFLAGS='-O -DNDEBUG'
            CDEBUGFLAGS='-g'
            CLIBS="$CLIBS -lstdc++"
#
            # C++keyName: nvcc
            # C++keyManufacturer: NVIDIA
            # C++keyLanguage: C++ CUDA
            # C++keyVersion:
            # C++keyLinkerName: GNU ld
            # C++keyLinkerVersion:
            # $NVCC was set above.
            CXX='$NVCC'
            CXXFLAGS=" -ccbin=gcc-4.4 -gencode=arch=compute_20,code=sm_20 -gencode=arch=compute_30,code=\\\"sm_30,compute_30\\\" -I$TMW_ROOT/toolbox/distcomp/gpu/extern/include --compiler-options=-ansi,-D_GNU_SOURCE,-fPIC,-fno-omit-frame-pointer,-pthread"
            CXXLIBS="$RPATH $MLIBS -lm -lmwgpu $TMW_ROOT/bin/$Arch/libcudart.so.5.0"
            CXXOPTIMFLAGS='-O -DNDEBUG'
            CXXDEBUGFLAGS='-g'
#
            LD="g++"
            LDEXTENSION='.mexa64'
            LDFLAGS="-pthread -shared -Wl,--version-script,$TMW_ROOT/extern/lib/$Arch/$MAPFILE -Wl,--no-undefined"
            LDOPTIMFLAGS='-O'
            LDDEBUGFLAGS='-g'
#
            POSTLINK_CMDS=':'
    esac
#############################################################################
#
# Architecture independent lines:
#
#     Set and uncomment any lines which will apply to all architectures.
#
#----------------------------------------------------------------------------
#           CC="$CC"
#           CFLAGS="$CFLAGS"
#           COPTIMFLAGS="$COPTIMFLAGS"
#           CDEBUGFLAGS="$CDEBUGFLAGS"
#           CLIBS="$CLIBS"
#
#           FC="$FC"
#           FFLAGS="$FFLAGS"
#           FOPTIMFLAGS="$FOPTIMFLAGS"
#           FDEBUGFLAGS="$FDEBUGFLAGS"
#           FLIBS="$FLIBS"
#
#           LD="$LD"
#           LDFLAGS="$LDFLAGS"
#           LDOPTIMFLAGS="$LDOPTIMFLAGS"
#           LDDEBUGFLAGS="$LDDEBUGFLAGS"
#----------------------------------------------------------------------------
#############################################################################
