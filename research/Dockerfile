FROM rapidsai/rapidsai-dev:0.14-cuda10.0-devel-ubuntu18.04-py3.6

# ARG http_proxy
# ARG https_proxy
# ARG no_proxy="${no_proxy}"
# ARG socks_proxy
# ARG TZ="Asia/Ho_Chi_Minh"

# ENV TERM=xterm \
#     http_proxy=${http_proxy}   \
#     https_proxy=${https_proxy} \
#     no_proxy=${no_proxy} \
#     socks_proxy=${socks_proxy} \
#     LANG='C.UTF-8'  \
#     LC_ALL='C.UTF-8' \
#     TZ=${TZ}

# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
# RUN apt-get update && apt-get --no-install-recommends install -yq ca-certificates libglib2.0-0 poppler-utils supervisor libgl1 tzdata git libxrender1 ffmpeg libsm6 libxext6 xcb && \
#     ln -fs /usr/share/zoneinfo/${TZ} /etc/localtime && dpkg-reconfigure -f tzdata && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

# RUN pip install scipy sklearn opencv-python openpyxl Pillow tqdm SimpleITK
# RUN pip install jupyter matplotlib PrettyTable
# RUN pip install tensorflow-gpu==1.15.5

CMD ["bash"]
