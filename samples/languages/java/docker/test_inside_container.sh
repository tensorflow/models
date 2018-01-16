#!/bin/bash

set -ex

cd /examples/label_image
mvn compile

cd /examples/object_detection
mvn compile

cd /examples/training
mvn compile
