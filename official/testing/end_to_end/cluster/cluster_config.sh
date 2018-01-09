#!/bin/bash
# These variables will be used for Kubernetes cluster set up.
# For full available set, please see Kubernetes documentation at
# https://kubernetes.io/docs/getting-started-guides/gce/

KUBE_ROOT=$HOME/kubernetes
KUBE_GCE_ZONE=europe-west1-d
NUM_NODES=2
NODE_SIZE=n1-standard-8
NODE_DISK_SIZE=200GB
KUBE_GCE_INSTANCE_PREFIX=tf-models-cluster-$$

# Change these OS vars only if you know what you are doing; 
# if you change one, chances are, you have to modify all,
# including the setup_worker.sh script.
KUBE_OS_DISTRIBUTION=ubuntu
KUBE_MASTER_OS_DISTRIBUTION=ubuntu
KUBE_GCE_MASTER_PROJECT=ubuntu-os-cloud
KUBE_GCE_MASTER_IMAGE=ubuntu-1604-xenial-v20161130
KUBE_NODE_OS_DISTRIBUTION=ubuntu
KUBE_GCE_NODE_PROJECT=ubuntu-os-cloud
KUBE_GCE_NODE_IMAGE=ubuntu-1604-xenial-v20161130
KUBE_NODE_EXTRA_METADATA=startup-script=$CURR_DIR/setup_worker.sh

USE_GPU=${USE_GPU:-true}
if [ "$USE_GPU" = true ] ; then
  # With GPU
  GPUS_PER_WORKER=8
  NODE_ACCELERATORS=type=nvidia-tesla-k80,count="${GPUS_PER_WORKER}"
  KUBE_FEATURE_GATES=Accelerators=true
else
  GPUS_PER_WORKER=
  NODE_ACCELERATORS=
  KUBE_FEATURE_GATES=
fi
