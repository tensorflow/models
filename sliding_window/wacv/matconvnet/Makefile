# file: Makefile
# author: Andrea Vedaldi
# brief: matconvnet makefile for mex files

# Copyright (C) 2014-16 Andrea Vedaldi
# All rights reserved.
#
# This file is part of the VLFeat library and is made available under
# the terms of the BSD license (see the COPYING file).

# ENABLE_GPU -- Set to YES to enable GPU support (requires CUDA and the MATLAB Parallel Toolbox)
# ENABLE_CUDNN -- Set to YES to enable CUDNN support. This will also likely require
# ENABLE_IMREADJPEG -- Set to YES to compile the function VL_IMREADJPEG()

ENABLE_GPU ?=
ENABLE_CUDNN ?=
ENABLE_IMREADJPEG ?= yes
ENABLE_DOUBLE ?= yes
DEBUG ?=
ARCH ?= maci64

# Configure MATLAB
MATLABROOT ?= /Applications/MATLAB_R2017a.app

# Configure CUDA and CuDNN. CUDAMETHOD can be either 'nvcc' or 'mex'.
CUDAROOT ?= /Developer/NVIDIA/CUDA-8.0
CUDNNROOT ?= $(CURDIR)/local/
CUDAMETHOD ?= $(if $(ENABLE_CUDNN),nvcc,mex)

# Remark: each MATLAB version requires a particular CUDA Toolkit version.
# Note that multiple CUDA Toolkits can be installed.
#MATLABROOT ?= /Applications/MATLAB_R2014b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-6.0
#MATLABROOT ?= /Applications/MATLAB_R2015a.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-7.0
#MATLABROOT ?= /Applications/MATLAB_R2015b.app
#CUDAROOT ?= /Developer/NVIDIA/CUDA-7.5

# Maintenance
NAME = matconvnet
VER = 1.0-beta24
DIST = $(NAME)-$(VER)
LATEST = $(NAME)-latest
RSYNC = rsync
HOST = vlfeat-admin:sites/sandbox-matconvnet
GIT = git
SHELL = /bin/bash # sh not good enough

# --------------------------------------------------------------------
#                                                        Configuration
# --------------------------------------------------------------------

# General options
MEX = $(MATLABROOT)/bin/mex
MEXEXT = $(MATLABROOT)/bin/mexext
MEXARCH = $(subst mex,,$(shell $(MEXEXT)))
MEXOPTS ?= matlab/src/config/mex_CUDA_$(ARCH).xml
NVCC = $(CUDAROOT)/bin/nvcc

comma:=,
space:=
space+=
join-with = $(subst $(space),$1,$(strip $2))
nvcc-quote = $(if $(strip $1),-Xcompiler $(call join-with,$(comma),$(1)),)

# Flags:
# 1.   `CXXFLAGS`: passed to `mex` and `nvcc` compiler wrappers
# 2.   `CXXFLAGS_PASS`: passed directly to the underlying C++ compiler
# 3.   `LDFLAGS`: passed directly to the underlying C++ compiler for linking
# 4.   `CXXOPTIMFLAGS`: passed directyl to the underlying C++ compiler
# 5.   `LDOPTIMFLAGS`: passed directly to the underlying C++ compiler
# 6.   `NVCCFLAGS_PASS`: passed directly to `nvcc` when invoked directly or through `mex`

CXXFLAGS = \
$(if $(ENABLE_GPU),-DENABLE_GPU,) \
$(if $(ENABLE_CUDNN),-DENABLE_CUDNN -I$(CUDNNROOT)/include,) \
$(if $(ENABLE_DOUBLE),-DENABLE_DOUBLE,) \
$(if $(VERB),-v,)
CXXFLAGS_PASS =
CXXOPTIMFLAGS =
LDFLAGS =
LDOPTIMFLAGS =
LINKLIBS = -lmwblas

NVCCFLAGS_PASS = -D_FORCE_INLINES -gencode=arch=compute_30,code=\"sm_30,compute_30\"
NVCCVER = $(shell $(NVCC) --version | \
sed -n 's/.*V\([0-9]*\).\([0-9]*\).\([0-9]*\).*/\1 \2 \3/p' | \
xargs printf '%02d%02d%02d')
NVCCVER_LT_70 = $(shell test $(NVCCVER) -lt 070000 && echo true)

# Mac OS X
ifeq "$(ARCH)" "$(filter $(ARCH),maci64)"
IMAGELIB ?= $(if $(ENABLE_IMREADJPEG),quartz,none)
CXXFLAGS_PASS += -mmacosx-version-min=10.9
CXXOPTIMFLAGS += -mssse3 -ffast-math
LDFLAGS += \
-mmacosx-version-min=10.9 \
$(if $(ENABLE_GPU),-Wl$(comma)-rpath -Wl$(comma)"$(CUDAROOT)/lib") \
$(if $(ENABLE_CUDNN),-Wl$(comma)-rpath -Wl$(comma)"$(CUDNNROOT)/lib") \
$(if $(NVCCVER_LT_70),-stdlib=libstdc++)
LINKLIBS += \
$(if $(ENABLE_GPU),-L"$(CUDAROOT)/lib" -lmwgpu -lcudart -lcublas) \
$(if $(ENABLE_CUDNN),-L"$(CUDNNROOT)/lib" -lcudnn)
endif

# Linux
ifeq "$(ARCH)" "$(filter $(ARCH),glnxa64)"
IMAGELIB ?= $(if $(ENABLE_IMREADJPEG),libjpeg,none)
CXXOPTIMFLAGS += -mssse3 -ftree-vect-loop-version -ffast-math -funroll-all-loops
LDFLAGS += \
$(if $(ENABLE_GPU),-Wl$(comma)-rpath -Wl$(comma)"$(CUDAROOT)/lib64") \
$(if $(ENABLE_CUDNN),-Wl$(comma)-rpath -Wl$(comma)"$(CUDNNROOT)/lib64")
LINKLIBS += \
-lrt \
$(if $(ENABLE_GPU),-L"$(CUDAROOT)/lib64" -lmwgpu -lcudart -lcublas) \
$(if $(ENABLE_CUDNN),-L"$(CUDNNROOT)/lib64" -lcudnn)
endif

# Image library
ifeq ($(IMAGELIB),libjpeg)
LINKLIBS += -ljpeg
endif
ifeq ($(IMAGELIB),quartz)
LINKLIBS += -framework Cocoa -framework ImageIO
endif

MEXFLAGS = $(CXXFLAGS) -largeArrayDims

ifneq ($(DEBUG),)
MEXFLAGS += -g
NVCCFLAGS_PASS += -g -O0
else
MEXFLAGS += -DNDEBUG -O
NVCCFLAGS_PASS += -DNDEBUG -O3
# include debug symbol also in non-debug version
CXXOPTIMFLAGS += -g
LDOPTIMFLAGS += -g
NVCCFLAGS_PASS += -g
endif

# --------------------------------------------------------------------
#                                                      Build MEX files
# --------------------------------------------------------------------

nvcc_filter=2> >( sed 's/^\(.*\)(\([0-9][0-9]*\)): \([ew].*\)/\1:\2: \3/g' >&2 )
cpp_src :=
mex_src :=

# Files that are compiled as CPP or GPU (CUDA) depending on whether GPU support
# is enabled.
ext := $(if $(ENABLE_GPU),cu,cpp)
cpp_src+=matlab/src/bits/data.$(ext)
cpp_src+=matlab/src/bits/datamex.$(ext)
cpp_src+=matlab/src/bits/nnconv.$(ext)
cpp_src+=matlab/src/bits/nnbias.$(ext)
cpp_src+=matlab/src/bits/nnfullyconnected.$(ext)
cpp_src+=matlab/src/bits/nnsubsample.$(ext)
cpp_src+=matlab/src/bits/nnpooling.$(ext)
cpp_src+=matlab/src/bits/nnnormalize.$(ext)
cpp_src+=matlab/src/bits/nnbnorm.$(ext)
cpp_src+=matlab/src/bits/nnbilinearsampler.$(ext)
cpp_src+=matlab/src/bits/nnroipooling.$(ext)
mex_src+=matlab/src/vl_nnconv.$(ext)
mex_src+=matlab/src/vl_nnconvt.$(ext)
mex_src+=matlab/src/vl_nnpool.$(ext)
mex_src+=matlab/src/vl_nnnormalize.$(ext)
mex_src+=matlab/src/vl_nnbnorm.$(ext)
mex_src+=matlab/src/vl_nnbilinearsampler.$(ext)
mex_src+=matlab/src/vl_nnroipool.$(ext)
mex_src+=matlab/src/vl_taccummex.$(ext)
mex_src+=matlab/src/vl_tmove.$(ext)
ifdef ENABLE_IMREADJPEG
mex_src+=matlab/src/vl_imreadjpeg.$(ext)
mex_src+=matlab/src/vl_imreadjpeg_old.$(ext)
endif

# CPU-specific files
cpp_src+=matlab/src/bits/impl/im2row_cpu.cpp
cpp_src+=matlab/src/bits/impl/subsample_cpu.cpp
cpp_src+=matlab/src/bits/impl/copy_cpu.cpp
cpp_src+=matlab/src/bits/impl/pooling_cpu.cpp
cpp_src+=matlab/src/bits/impl/normalize_cpu.cpp
cpp_src+=matlab/src/bits/impl/bnorm_cpu.cpp
cpp_src+=matlab/src/bits/impl/bilinearsampler_cpu.cpp
cpp_src+=matlab/src/bits/impl/roipooling_cpu.cpp
cpp_src+=matlab/src/bits/impl/tinythread.cpp
ifdef ENABLE_IMREADJPEG
cpp_src+=matlab/src/bits/impl/imread_$(IMAGELIB).cpp
cpp_src+=matlab/src/bits/imread.cpp
endif

# GPU-specific files
ifdef ENABLE_GPU
cpp_src+=matlab/src/bits/impl/im2row_gpu.cu
cpp_src+=matlab/src/bits/impl/subsample_gpu.cu
cpp_src+=matlab/src/bits/impl/copy_gpu.cu
cpp_src+=matlab/src/bits/impl/pooling_gpu.cu
cpp_src+=matlab/src/bits/impl/normalize_gpu.cu
cpp_src+=matlab/src/bits/impl/bnorm_gpu.cu
cpp_src+=matlab/src/bits/impl/bilinearsampler_gpu.cu
cpp_src+=matlab/src/bits/impl/roipooling_gpu.cu
cpp_src+=matlab/src/bits/datacu.cu
mex_src+=matlab/src/vl_cudatool.cu
ifdef ENABLE_CUDNN
cpp_src+=matlab/src/bits/impl/nnconv_cudnn.cu
cpp_src+=matlab/src/bits/impl/nnpooling_cudnn.cu
cpp_src+=matlab/src/bits/impl/nnbias_cudnn.cu
cpp_src+=matlab/src/bits/impl/nnbilinearsampler_cudnn.cu
cpp_src+=matlab/src/bits/impl/nnbnorm_cudnn.cu
endif
endif

mex_tgt:=$(patsubst %.cpp,%.mex$(MEXARCH),$(mex_src))
mex_tgt:=$(patsubst %.cu,%.mex$(MEXARCH),$(mex_tgt))
mex_tgt:=$(subst matlab/src/,matlab/mex/,$(mex_tgt))

mex_obj:=$(patsubst %.cpp,%.o,$(mex_src))
mex_obj:=$(patsubst %.cu,%.o,$(mex_obj))
mex_obj:=$(subst matlab/src/,matlab/mex/.build/,$(mex_obj))

cpp_tgt:=$(patsubst %.cpp,%.o,$(cpp_src))
cpp_tgt:=$(patsubst %.cu,%.o,$(cpp_tgt))
cpp_tgt:=$(subst matlab/src/,matlab/mex/.build/,$(cpp_tgt))

.PHONY: all, distclean, clean, info, pack, post, post-doc, doc

all: $(cpp_tgt) $(mex_obj) $(mex_tgt)

# Create build directory
%/.stamp:
	mkdir -p $(*)/ ; touch $(*)/.stamp
$(mex_tgt): matlab/mex/.build/bits/impl/.stamp
$(cpp_tgt): matlab/mex/.build/bits/impl/.stamp

# Standard code
.PRECIOUS: matlab/mex/.build/%.o
.PRECIOUS: %/.stamp

matlab/mex/.build/bits/impl/imread.o : matlab/src/bits/impl/imread_helpers.hpp
matlab/mex/.build/bits/impl/imread_quartz.o : matlab/src/bits/impl/imread_helpers.hpp
matlab/mex/.build/bits/impl/imread_gdiplus.o : matlab/src/bits/impl/imread_helpers.hpp
matlab/mex/.build/bits/impl/imread_libjpeg.o : matlab/src/bits/impl/imread_helpers.hpp

# --------------------------------------------------------------------
#                                                    Compilation rules
# --------------------------------------------------------------------

MEXFLAGS_CC_CPU := \
$(MEXFLAGS) \
CXXFLAGS='$$CXXFLAGS $(CXXFLAGS_PASS)' \
CXXOPTIMFLAGS='$$CXXOPTIMFLAGS $(CXXOPTIMFLAGS)'

MEXFLAGS_CC_GPU := \
-f "$(MEXOPTS)" \
$(MEXFLAGS) \
CXXFLAGS='$$CXXFLAGS $(NVCCFLAGS_PASS) $(call nvcc-quote,$(CXXFLAGS_PASS))' \
CXXOPTIMFLAGS='$$CXXOPTIMFLAGS $(call nvcc-quote,$(CXXOPTIMFLAGS))'

MEXFLAGS_LD := $(MEXFLAGS) \
LDFLAGS='$$LDFLAGS $(LDFLAGS)' \
LDOPTIMFLAGS='$$LDOPTIMFLAGS $(LDOPTIMFLAGS)' \
LINKLIBS='$(LINKLIBS) $$LINKLIBS' \

NVCCFLAGS = $(CXXFLAGS) $(NVCCFLAGS_PASS) \
-I"$(MATLABROOT)/extern/include" \
-I"$(MATLABROOT)/toolbox/distcomp/gpu/extern/include" \
$(call nvcc-quote,-fPIC $(CXXFLAGS_PASS) $(CXXOPTIMFLAGS))

ifneq ($(ENABLE_GPU),)
ifeq ($(CUDAMETHOD),mex)
matlab/mex/.build/%.o : matlab/src/%.cu matlab/mex/.build/.stamp
	MW_NVCC_PATH='$(NVCC)' \
	$(MEX) -c $(MEXFLAGS_CC_GPU) "$(<)" $(nvcc_filter)
	mv -f "$(notdir $(@))" "$(@)"
else
matlab/mex/.build/%.o : matlab/src/%.cu matlab/mex/.build/.stamp
	$(NVCC) $(NVCCFLAGS) "$(<)" -c -o "$(@)" $(nvcc_filter)
endif
endif

matlab/mex/.build/%.o : matlab/src/%.cpp matlab/src/%.cu matlab/mex/.build/.stamp
	$(MEX) -c $(MEXFLAGS_CC_CPU) "$(<)"
	mv -f "$(notdir $(@))" "$(@)"

matlab/mex/.build/%.o : matlab/src/%.cpp matlab/mex/.build/.stamp
	$(MEX) -c $(MEXFLAGS_CC_CPU) "$(<)"
	mv -f "$(notdir $(@))" "$(@)"

matlab/mex/%.mex$(MEXARCH) : matlab/mex/.build/%.o $(cpp_tgt)
	$(MEX) $(MEXFLAGS_LD) "$(<)" -output "$(@)" $(cpp_tgt)

# --------------------------------------------------------------------
#                                                        Documentation
# --------------------------------------------------------------------

include doc/Makefile

# --------------------------------------------------------------------
#                                                          Maintenance
# --------------------------------------------------------------------

info: doc-info
	@echo "mex_src=$(mex_src)"
	@echo "mex_obj=$(mex_obj)"
	@echo "mex_tgt=$(mex_tgt)"
	@echo "cpp_src=$(cpp_src)"
	@echo "cpp_tgt=$(cpp_tgt)"
	@echo '------------------------------'
	@echo 'CUDAMETHOD=$(CUDAMETHOD)'
	@echo 'CXXFLAGS=$(CXXFLAGS)'
	@echo 'CXXOPTIMFLAGS=$(CXXOPTIMFLAGS)'
	@echo 'LDFLAGS=$(LDFLAGS)'
	@echo 'LDOPTIMFLAGS=$(LDOPTIMFLAGS)'
	@echo 'LINKLIBS=$(LINKLIBS)'
	@echo '------------------------------'
	@echo 'MEXARCH=$(MEXARCH)'
	@echo 'MEXFLAGS=$(MEXFLAGS)'
	@echo 'MEXFLAGS_CC_CPU=$(MEXFLAGS_CC_CPU)'
	@echo 'MEXFLAGS_CC_GPU=$(MEXFLAGS_CC_GPU)'
	@echo 'MEXFLAGS_LD=$(MEXFLAGS_LD)'
	@echo '------------------------------'
	@echo 'NVCC=$(NVCC)'
	@echo 'NVCCVER=$(NVCCVER)'
	@echo 'NVCCVER_LT_70=$(NVCCVER_LT_70)'
	@echo 'NVCCFLAGS_PASS=$(NVCCFLAGS_PASS)'
	@echo 'NVCCFLAGS=$(NVCCFLAGS)'


clean: doc-clean
	find . -name '*~' -delete
	rm -f $(cpp_tgt)
	rm -rf matlab/mex/.build

distclean: clean doc-distclean
	rm -rf matlab/mex
	rm -f doc/index.html doc/matconvnet-manual.pdf
	rm -f $(NAME)-*.tar.gz

pack:
	COPYFILE_DISABLE=1 \
	COPY_EXTENDED_ATTRIBUTES_DISABLE=1 \
	$(GIT) archive --prefix=$(NAME)-$(VER)/ v$(VER) | gzip > $(DIST).tar.gz
	ln -sf $(DIST).tar.gz $(LATEST).tar.gz

post: pack
	$(RSYNC) -aP $(DIST).tar.gz $(LATEST).tar.gz $(HOST)/download/

post-models:
	$(RSYNC) -aP data/models/*.mat $(HOST)/models/

post-doc: doc
	$(RSYNC) -aP doc/matconvnet-manual.pdf $(HOST)/
	$(RSYNC) \
		--recursive \
		--perms \
	        --verbose \
	        --delete \
	        --exclude=download \
	        --exclude=models \
	        --exclude=matconvnet-manual.pdf \
	        --exclude=.htaccess doc/site/site/ $(HOST)/

.PHONY: model-md5
model-md5:
	cd data/models ; md5sum *.mat | xargs  printf '| %-33s| %-40s|\n'
