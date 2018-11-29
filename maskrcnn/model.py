import numpy as np

import tensorflow as tf
if tf.__version__ == '1.5.0':
    import keras
else:
    from tensorflow import keras

from .subgraphs.fpn_backbone_graph import BackboneGraph
from .subgraphs.rpn_graph import RPNGraph
from .subgraphs.proposal_layer import ProposalLayer
from .subgraphs.fpn_classifier_graph import FPNClassifierGraph
from .subgraphs.fpn_mask_graph import FPNMaskGraph
from .subgraphs.detection_layer import DetectionLayer

from .results import _results_from_tensor_values

def _build_keras_models(mode):

    assert mode in ["tf.estimator", "coreml"]

    architecture = 'resnet101'
    input_width = 1024
    input_height = 1024
    input_image_shape = (input_width, input_height)
    num_classes = 1 + 80
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

    input_image = keras.layers.Input(shape=[input_width, input_height, 3], name="input_image")

    backbone = BackboneGraph(input_tensor=input_image,
                             architecture=architecture,
                             pyramid_size=pyramid_top_down_size)

    P2, P3, P4, P5, P6 = backbone.build()

    rpn = RPNGraph(anchor_stride=anchor_stride,
                   anchors_per_location=anchors_per_location,
                   depth=pyramid_top_down_size,
                   feature_maps=[P2, P3, P4, P5, P6])

    # anchor_object_probs: Probability of each anchor containing only background or objects
    # anchor_deltas: Bounding box refinements to apply to each anchor to better enclose its object
    anchor_object_probs, anchor_deltas = rpn.build()

    # rois: Regions of interest (regions of the image that probably contain an object)
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
                                              pyramid_top_down_size=pyramid_top_down_size)

    # rois_class_probs: Probability of each class being contained within the roi
    # rois_deltas: Bounding box refinements to apply to each roi to better enclose its object
    fpn_classifier_model, classification = fpn_classifier_graph.build()

    detections = DetectionLayer(name="detections",
                                max_detections=max_detections,
                                bounding_box_std_dev=bounding_box_std_dev,
                                detection_min_confidence=detection_min_confidence,
                                detection_nms_threshold=detection_nms_threshold)([rois, classification])

    if mode == "coreml":
        #TODO: eventually remove this useless operation, but now required for CoreML
        detections = keras.layers.Reshape((max_detections, 6))(detections)

    fpn_mask_graph = FPNMaskGraph(rois=detections,
                                  feature_maps=mrcnn_feature_maps,
                                  pool_size=mask_pool_size,
                                  image_shape=input_image_shape,
                                  num_classes=num_classes,
                                  max_regions=max_detections,
                                  pyramid_top_down_size=pyramid_top_down_size)

    fpn_mask_model, masks = fpn_mask_graph.build()

    mask_rcnn_model = keras.models.Model(input_image,
                                         [detections, masks],
                                         name='mask_rcnn_model')

    return mask_rcnn_model, fpn_classifier_model, fpn_mask_model, proposal_layer.anchors

class MaskRCNNModel():

    _estimator = None

    def __init__(self,
                 config_path,
                 model_dir=None,
                 run_config=None,
                 initial_keras_weights=None):

        self.config_path = config_path#TODO: actually read the config instead
        self.model_dir = model_dir
        self.run_config = run_config
        self.initial_keras_weights = initial_keras_weights

    def train(self,
              input_fn,
              steps=None,
              max_steps=None):
        estimator = self._get_estimator()
        return estimator.train(input_fn, steps=steps, max_steps=max_steps)

    def evaluate(self,
                 input_fn,
                 steps=None):
        estimator = self._get_estimator()
        metrics = estimator.evaluate(input_fn, steps=steps)
        return metrics

    def train_and_evaluate(self,
                           train_input_fn,
                           eval_input_fn,
                           train_steps=None,
                           max_train_steps=None,
                           eval_steps=None):
        self.train(train_input_fn, steps=train_steps, max_steps=max_train_steps)
        return self.evaluate(eval_input_fn, steps=eval_steps)

    def predict(self,
                dataset_id,
                input_fn,
                image_info_fn,
                class_label_fn):
        estimator = self._get_estimator()
        tensor_values = estimator.predict(input_fn)
        return _results_from_tensor_values(tensor_values,
                                           dataset_id=dataset_id,
                                           image_info_fn=image_info_fn,
                                           class_label_fn=class_label_fn)

    def get_trained_keras_models(self):

        mask_rcnn_model, \
        fpn_classifier_model, \
        fpn_mask_model, \
        anchors = self._build_keras_models(environment="coreml")

        checkpoint = self._get_checkpoint()
        if checkpoint:
            #TODO: convert to keras weights
            #TODO: assign the weights to all relevant layers
            pass
        else:
            #Otherwise we load the weights
            assert self.initial_keras_weights is not None

            mask_rcnn_model.load_weights(self.initial_keras_weights, by_name=True)
            fpn_classifier_model.load_weights(self.initial_keras_weights, by_name=True)
            fpn_mask_model.load_weights(self.initial_keras_weights, by_name=True)

        return mask_rcnn_model, fpn_classifier_model, fpn_mask_model, anchors

    def export_estimator(self):
        # TODO:
        pass

    def _get_estimator(self):
        if self._estimator is None:
            self._estimator = self._build_estimator()
        return self._estimator

    def _get_checkpoint(self):
        #TODO: get the checkpoint from model_dir
        return None

    def _build_keras_models(self, environment):
        #TODO: extract the config
        return _build_keras_models(environment)

    def _build_estimator(self):
        #TODO: we might want to skip this and load the model from the model_dir?
        mask_rcnn_model, _, _, _ = self._build_keras_models(environment = "tf.estimator")

        #TODO: only load the weights if we do not have a checkpoint?
        if self.initial_keras_weights:
            mask_rcnn_model.load_weights(self.initial_keras_weights, by_name=True)

        optimizer = keras.optimizers.SGD(
            lr=0.001, momentum=0.9,
            clipnorm=5.0)

        optimizer = _CustomOptimizer()

        def custom_loss(y_true, y_pred):
            loss = keras.backend.constant(0)
            loss = keras.backend.stop_gradient(loss)
            return loss

        def mAP(y_true, y_pred):
            #TODO: associate to a dynamic metric
            print(y_true)
            print(y_pred)
            return keras.backend.constant(0)

        mask_rcnn_model.compile(
            optimizer=optimizer,
            loss=[custom_loss, custom_loss],
            metrics=[mAP])

        return keras.estimator.model_to_estimator(mask_rcnn_model, model_dir=self.model_dir)

    #TEMPORARY
class _CustomOptimizer(keras.optimizers.Optimizer):
    def get_updates(self, loss, params):
        return []