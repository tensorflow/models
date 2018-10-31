# Minigo Testing

This directory contains test infrastructure for Minigo, largely based of the
work done by https://github.com/kubeflow/kubeflow.

Our tests are run on the Kubernetes test runner called prow. See the [Prow
docs](https://github.com/kubernetes/test-infra/tree/master/prow) for more
details.

Some UIs to check out:

Testgrid (Test Results Dashboard): https://k8s-testgrid.appspot.com/sig-big-data
Prow (Test-runner dashboard): https://prow.k8s.io/?repo=tensorflow%2Fminigo

## Local testing

To test out changes to the docker image, first build the test-harness image:

```shell
make buildv2
```

And then run the tests.

```shell
docker run --rm gcr.io/minigo-testing/minigo-prow-harness-v2:latest --repo=github.com/tensorflow/minigo --job=tf-minigo-presubmit
```

## Prow configuration

Minigo has some configuration directly in Prow to make all this jazz work:

- Test configuration:
  https://github.com/kubernetes/test-infra/blob/master/prow/config.yaml
- Test UI Configuration:
  https://github.com/kubernetes/test-infra/blob/master/testgrid/config.yaml
- Bootstrap-jobs-config:
  https://github.com/kubernetes/test-infra/blob/master/jobs/config.json
