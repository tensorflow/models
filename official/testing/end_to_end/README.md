## End-to-End Testing of Official Models

In order to facilitate performance and convergence testing of the official Model Garden models, we provide a set of scripts and tools that can:

1. Spin up a Kubernetes cluster on Google Cloud Platform.
2. Deploy jobs to the cluster.
3. Run performance and convergence tests for Garden models.
4. Produce results for easy consumption or dashboard creation.

Want to run your own cluster and tests? Read on!

### Pre-requisites

Note: Throughout this guide, we will assume a working knowledge of [Kubernetes](https://kubernetes.io/) (k8s) and [Google Cloud Platform](https://cloud.google.com) (GCP). If you are new to either, please run through the [appropriate tutorials](https://kubernetes.io/docs/getting-started-guides/gce/) first. 

Before launching any testing jobs, you will need to configure your local environment so as to be able to access both k8s and GCP. Here, we will briefly walk you through the necessary installation steps, although further documentation should be obtained from k8s and GCP websites directly.

#### Install gcloud and kubectl

You will need [gcloud](https://cloud.google.com/sdk/gcloud/) and [kubectl](https://kubernetes.io/docs/reference/kubectl/overview/) installed on your local machine in order to deploy and monitor jobs. Please follow the [Google Cloud SDK installation instructions](https://cloud.google.com/sdk/docs/) and the [kubectl installation instructions](https://kubernetes.io/docs/tasks/tools/install-kubectl/).
