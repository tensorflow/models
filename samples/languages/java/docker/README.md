Dockerfile for building an image suitable for running the Java examples.

Typical usage:

```
docker build -t java-tensorflow .
docker run -it --rm -v ${PWD}/..:/examples -w /examples java-tensorflow
```

That second command will pop you into a shell which has all
the dependencies required to execute the scripts and Java
examples.

The script `sanity_test.sh` builds this container and runs a compilation
check on all the maven projects.
