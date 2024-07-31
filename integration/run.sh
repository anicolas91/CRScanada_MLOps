#!/bin/bash
cd "$(dirname "$0")"

# since we are simulating s3 via localstack, we are just going to make sure the image/container runs.
# composing down and again up would lose the data so we just use the already running one.

DOCKER_IMAGE_NAME="crs_score_prediction:v1"

# Check if the container with the given image is running
CONTAINER_ID=$(docker ps -q -f "ancestor=$DOCKER_IMAGE_NAME")

if [ -z "$CONTAINER_ID" ]; then
    echo "No container is running for the image: $DOCKER_IMAGE_NAME"
    exit 1
fi

echo "Container with image $DOCKER_IMAGE_NAME is running (ID: $CONTAINER_ID)."

python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} -eq 0 ]; then
    echo "Python script executed successfully."
else
    echo "Failed to execute Python script."
    docker logs $CONTAINER_ID
    exit ${ERROR_CODE}
fi