#!/bin/bash

set -e
if [[ $# -eq 0 ]] || [ "$1" == "-h" ] ; then
  echo "usage: ($0) [image_name]"
  exit 0
fi

# set image name
IMAGE_NAME_ARG=$1
PROJECT_ID=$(gcloud config get-value core/project)
IMAGE_VERSION=latest
IMAGE_NAME=gcr.io/$PROJECT_ID/$IMAGE_NAME_ARG
GCP_IMAGE="${IMAGE_NAME}":$IMAGE_VERSION

# Building docker image.
docker build -t "${GCP_IMAGE}" -f kfp/components/upload_model/Dockerfile .

# Push docker image
gcloud auth configure-docker --quiet
docker push "${GCP_IMAGE}"

# Output the strict image name (which contains the sha256 image digest)
docker inspect --format="{{index .RepoDigests 0}}" "${GCP_IMAGE}"
