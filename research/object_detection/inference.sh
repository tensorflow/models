bazel build -c opt --config=opt --config=cuda --copt=-mavx --copt=-mavx2 --copt=-mfma --copt=-mfpmath=both --copt=-msse4.2 --config=cuda -k --verbose_failures --crosstool_top=@local_config_cuda//crosstool:toolchain --spawn_strategy=standalone --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
  research/object_detection:inference_on_image

bazel-bin/research/object_detection/inference_on_image
