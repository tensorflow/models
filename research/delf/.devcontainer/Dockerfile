#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------
ARG IMAGE_TYPE
ARG TF_VERSION="latest"
FROM tensorflow/tensorflow:$TF_VERSION$IMAGE_TYPE-jupyter

# This Dockerfile adds a non-root user with sudo access. Update the “remoteUser” property in
# devcontainer.json to use it. More info: https://aka.ms/vscode-remote/containers/non-root-user.
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Options for common setup script - SHA updated on release
ARG INSTALL_ZSH="true"
ARG UPGRADE_PACKAGES="false"
ARG COMMON_SCRIPT_SOURCE="https://raw.githubusercontent.com/microsoft/vscode-dev-containers/main/script-library/common-debian.sh"
ARG COMMON_SCRIPT_SHA="dev-mode"

# Install needed packages and setup non-root user. Use a separate RUN statement to add your own dependencies.
RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends curl ca-certificates 2>&1 \
    && curl -sSL  ${COMMON_SCRIPT_SOURCE} -o /tmp/common-setup.sh \
    && ([ "${COMMON_SCRIPT_SHA}" = "dev-mode" ] || (echo "${COMMON_SCRIPT_SHA} */tmp/common-setup.sh" | sha256sum -c -)) \
    && /bin/bash /tmp/common-setup.sh "${INSTALL_ZSH}" "${USERNAME}" "${USER_UID}" "${USER_GID}" "${UPGRADE_PACKAGES}" \
    && rm /tmp/common-setup.sh \
    #
    # *********************************************************************
    # * Uncomment this section to use RUN to install other dependencies.  *
    # * See https://aka.ms/vscode-remote/containers/dockerfile-run        *
    # *********************************************************************
    && apt-get -y install --no-install-recommends protobuf-compiler \
    #
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

COPY object_detection/ /tmp/od/object_detection/
RUN cd /tmp/od/ && protoc object_detection/protos/*.proto --python_out=. \
    && cp /tmp/od/object_detection/packages/tf2/setup.py /tmp/od  \
    && pip3 --disable-pip-version-check --no-cache-dir install --use-feature=2020-resolver . \
    && rm -rf /tmp/od

RUN pip3 --disable-pip-version-check --no-cache-dir install matplotlib numpy scikit-image scipy
