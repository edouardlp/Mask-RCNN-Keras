#!/bin/sh
docker build -t mask_rcnn_eval -f examples/coco/Dockerfile .
docker run -it \
--rm \
--name mask_rcnn_eval \
--mount type=bind,source=/Users/elaveryplante/Documents/Development/Mask-RCNN-CoreML/Data,target=/usr/src/app/Data \
mask_rcnn_eval
