# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file builds the images that are used by prow to run the Minigo tests.

IMG = gcr.io/minigo-testing/minigo-prow-harness
IMG_V2 = gcr.io/minigo-testing/minigo-prow-harness-v2
TAG := $(shell date +v%Y%m%d)-$(shell git describe --tags --always --dirty)-$(shell git diff | openssl sha -sha256 | cut -c -6)
DIR := ${CURDIR}

all: build

build:
	mkdir -p staging
	cp ../requirements.txt staging/requirements.txt
	@echo {\"image\": \"$(IMG):$(TAG)\"} > version.json
	docker build ${DOCKER_BUILD_OPTS} -t $(IMG):$(TAG) .
	docker tag $(IMG):$(TAG) $(IMG):latest
	rm -R staging/*
	rmdir staging
	@echo Built $(IMG):$(TAG) and tagged with latest

push: build
	gcloud docker -- push $(IMG):$(TAG)
	gcloud docker -- push $(IMG):latest
	@echo Pushed $(IMG) with :latest and :$(TAG) tags

buildv2:
	mkdir -p staging
	cp ../requirements.txt staging/requirements.txt
	@echo {\"image\": \"$(IMG_V2):$(TAG)\"} > version-v2.json
	docker build -f Dockerfile.v2 -t $(IMG_V2):$(TAG) .
	docker tag $(IMG_V2):$(TAG) $(IMG_V2):latest
	rm -R staging/*
	rmdir staging
	@echo Built $(IMG_V2):$(TAG) and tagged with latest

pushv2: buildv2
	gcloud docker -- push $(IMG_V2):$(TAG)
	gcloud docker -- push $(IMG_V2):latest
	@echo Pushed $(IMG_V2) with :latest and :$(TAG) tags
