import os
import math
import json
import keras
import numpy as np
import json

from .subgraphs.fpn_backbone_graph import BackboneGraph
from .subgraphs.rpn_graph import RPNGraph
from .subgraphs.proposal_layer import ProposalLayer
from .subgraphs.fpn_classifier_graph import FPNClassifierGraph
from .subgraphs.fpn_mask_graph import FPNMaskGraph
from .subgraphs.detection_layer import DetectionLayer

def build_models(config_path,
                 weights_path):

    #TODO: load from config_path

    architecture = 'resnet101'
    input_width = 1024
    input_height = 1024
    input_image_shape = (input_width,input_height)
    num_classes = 1+80
    pre_nms_max_proposals = 6000
    max_proposals = 1000
    max_detections = 100
    pyramid_top_down_size = 256
    proposal_nms_threshold = 0.7
    detection_min_confidence = 0.7
    detection_nms_threshold = 0.3
    bounding_box_std_dev = [0.1, 0.1, 0.2, 0.2]
    classifier_pool_size = 7
    mask_pool_size = 14
    fc_layers_size = 1024
    anchor_scales = (32, 64, 128, 256, 512)
    anchor_ratios = [0.5, 1, 2]
    anchors_per_location = len(anchor_ratios)
    backbone_strides = [4, 8, 16, 32, 64]
    anchor_stride = 1

    input_image = keras.layers.Input(shape=[input_width,input_height,3], name="input_image")

    backbone = BackboneGraph(input_tensor=input_image,
                             architecture = architecture,
                             pyramid_size = pyramid_top_down_size)

    P2, P3, P4, P5, P6 = backbone.build()

    rpn = RPNGraph(anchor_stride=anchor_stride,
                   anchors_per_location=anchors_per_location,
                   depth=pyramid_top_down_size,
                   feature_maps=[P2, P3, P4, P5, P6])

    #anchor_object_probs: Probability of each anchor containing only background or objects
    #anchor_deltas: Bounding box refinements to apply to each anchor to better enclose its object
    anchor_object_probs, anchor_deltas = rpn.build()

    #rois: Regions of interest (regions of the image that probably contain an object)
    proposal_layer = ProposalLayer(name="ROI",
                                   image_shape=input_image_shape,
                                   max_proposals=max_proposals,
                                   pre_nms_max_proposals=pre_nms_max_proposals,
                                   bounding_box_std_dev=bounding_box_std_dev,
                                   nms_threshold=proposal_nms_threshold,
                                   anchor_scales=anchor_scales,
                                   anchor_ratios=anchor_ratios,
                                   backbone_strides=backbone_strides,
                                   anchor_stride=anchor_stride)

    rois = proposal_layer([anchor_object_probs, anchor_deltas])

    mrcnn_feature_maps = [P2, P3, P4, P5]

    fpn_classifier_graph = FPNClassifierGraph(rois=rois,
                                              feature_maps=mrcnn_feature_maps,
                                              pool_size=classifier_pool_size,
                                              image_shape=input_image_shape,
                                              num_classes=num_classes,
                                              max_regions=max_proposals,
                                              fc_layers_size=fc_layers_size,
                                              pyramid_top_down_size=pyramid_top_down_size,
                                              weights_path=weights_path)

    #rois_class_probs: Probability of each class being contained within the roi
    #rois_deltas: Bounding box refinements to apply to each roi to better enclose its object
    fpn_classifier_model, classification = fpn_classifier_graph.build()


    detections = DetectionLayer(name="mrcnn_detection",
                                max_detections=max_detections,
                                bounding_box_std_dev=bounding_box_std_dev,
                                detection_min_confidence=detection_min_confidence,
                                detection_nms_threshold=detection_nms_threshold)([rois, classification])

    #TODO: try to remove this line (CoreML crashes without it)
    detections = keras.layers.Reshape((max_detections,6))(detections)

    fpn_mask_graph = FPNMaskGraph(rois=detections,
                                  feature_maps=mrcnn_feature_maps,
                                  pool_size=mask_pool_size,
                                  image_shape=input_image_shape,
                                  num_classes=num_classes,
                                  max_regions=max_detections,
                                  pyramid_top_down_size=pyramid_top_down_size,
                                  weights_path=weights_path)

    fpn_mask_model, masks = fpn_mask_graph.build()

    mask_rcnn_model = keras.models.Model(input_image,
                                     [detections,masks],
                                     name='mask_rcnn_model')

    mask_rcnn_model.load_weights(weights_path, by_name=True)
    return mask_rcnn_model, fpn_classifier_model, fpn_mask_model, proposal_layer.anchors

def predict(config_path,
            weights_path,
            results_path,
            image_ids,
            images,
            params):

    predict = True

    if predict:
        mask_rcnn_model, _, _,_ = build_models(config_path,weights_path)
        results = mask_rcnn_model.predict(images)
        print(results[0])
        np.save("detections.npy", results[0])
        np.save("masks.npy", results[1])

    detections = np.load("detections.npy")
    masks = np.load("masks.npy")
    print(masks.shape)
    masks = np.transpose(masks,axes=[0,3,2,1])
    detections = build_detections(image_ids,detections,masks)

    with open(results_path, 'w') as outfile:
        json.dump(detections, outfile)

def build_detections(image_ids,detections_array, masks_array):
    detections = []

    for i in range(0,detections_array.shape[0]):
        id = image_ids[i]
        for j in range(0,detections_array.shape[1]):
            detection = build_detection(detections_array[i, j, :])
            if detection:
                detection["image_id"] = id
                segmentation = rle_encoding(masks_array[i, j, :] > 0.5)
                detection["segmentation"] = segmentation
                detections.append(detection)
    return detections

def rle_encoding(x):
    '''
    x: numpy array of shape (height, width), 1 - mask, 0 - background
    Returns run length as list
    '''
    dots = np.where(x.T.flatten() == 1)[0]  # .T sets Fortran order down-then-right
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((int(b + 1), 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def build_detection(array):
    category_id = int(array[4])
    if(category_id == 0):
        return None
    score = float(array[5])
    bbox = array[0:4].tolist()
    return { "category_id":category_id ,"score":score, "bbox":  bbox}
