#!/bin/bash

CLUSTER_CONFIG=${CLUSTER_CONFIG:-cluster_config.sh}
set -o allexport
source $CLUSTER_CONFIG
set +o allexport

echo "Taking down existing cluster..."
$KUBE_ROOT/cluster/kube-down.sh

echo "Starting new cluster..."
$KUBE_ROOT/cluster/kube-up.sh
