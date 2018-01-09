## End-to-End Testing of Official Models

In order to facilitate performance and convergence testing of the official Model Garden models, we provide a set of scripts and tools that can:

1. Spin up a Kubernetes cluster on Google Cloud Platform.
2. Deploy jobs to the cluster.
3. Run performance and convergence tests for Garden models.
4. Produce results for easy consumption or dashboard creation.

Want to run your own cluster and tests? Read on!

### Pre-requisites

Note: Throughout this guide, we will assume a working knowledge of [Kubernetes](https://kubernetes.io/) (k8s) and [Google Cloud Platform](https://cloud.google.com) (GCP). If you are new to either, please run through the [appropriate tutorials](https://kubernetes.io/docs/getting-started-guides/gce/) first. (Implied in that: you will need access to a GCP account and project; if you don't have one, [sign up](https://console.cloud.google.com/start).)

Before launching any testing jobs, you will need to configure your local environment so as to be able to access both k8s and GCP. Here, we will briefly walk you through the necessary installation steps, although further documentation should be obtained from k8s and GCP websites directly.

#### Install gcloud and kubectl

You will need [gcloud](https://cloud.google.com/sdk/gcloud/) and [kubectl](https://kubernetes.io/docs/reference/kubectl/overview/) installed on your local machine in order to deploy and monitor jobs. Please follow the [Google Cloud SDK installation instructions](https://cloud.google.com/sdk/docs/) and the [kubectl installation instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl/).

After installation, make sure to set gcloud to point to the GCP project you will be using:

```
gcloud config set project <project-id>
```

#### Get a Kubernetes cluster

The tools in the workload directory create pods and jobs on an existing cluster-- which means you need access to a k8s cluster on GCP. The easiest way to do that is to use the [Google Kubernetes Engine](https://cloud.google.com/kubernetes-engine/) (GKE), which has a handy UI. However, GKE does not (as of January 2018) support GPU nodes, so the GKE solution will only work if you are running tests on CPU nodes. 

In any case, if you are using GKE and manually set up a cluster, or if you happen to already have access to a GCP k8s cluster previously created, you can skip the rest of this section on creating a k8s cluster, and instead just configure `gcloud` to access your cluster:

```
gcloud config set project <project-id>
gcloud config set region <region-id>
kubectl config use-context <cluster-auth-id>
```

If you don't have a cluster already, you can use the scripts in the cluster directory to set one up:

1. In your local copy of the cluster directory, edit the text file cluster/cluster_config.sh to reflect your desired configuration details. (It may be the case that you don't need to change anything at all. If you installed kubectl in a non-standard location, update `KUBE_ROOT`; if you want to use a different region, update `KUBE_GCE_ZONE`, though note that GPUs are only [available in certain regions](https://cloud.google.com/compute/docs/gpus/); if you don't want to use a GPU cluster, update `USE_GPU`; and if you want a different number of worker nodes, update `NUM_NODES`. Note that you should only change the operating system variables if you know what you are doing, as we do not currently support other variants.)

