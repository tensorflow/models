#!/bin/bash
# Loads configuration and runs Kubernetes cluster scripts.
# For full available set, please see Kubernetes documentation at
# https://kubernetes.io/docs/getting-started-guides/gce/

export CURR_DIR=`dirname $0`
CLUSTER_CONFIG=${CLUSTER_CONFIG:-$CURR_DIR/cluster_config.sh}
set -o allexport
source $CLUSTER_CONFIG
set +o allexport

echo "Taking down existing cluster..."
$KUBE_ROOT/cluster/kube-down.sh

echo "Starting new cluster..."
$KUBE_ROOT/cluster/kube-up.sh
