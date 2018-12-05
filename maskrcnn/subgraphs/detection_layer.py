import tensorflow as tf
if tf.__version__ == '1.5.0':
    import keras
    from keras.engine import Layer
    from tensorflow import sparse_tensor_to_dense as to_dense
else:
    from tensorflow import keras
    from tensorflow.keras.layers import Layer
    from tensorflow.sparse import to_dense

import numpy as np

from .utils import apply_box_deltas_graph
from .utils import clip_boxes_graph
from .utils import norm_boxes_graph

#NOTE: None of this will get exported to CoreML. This is only useful for python inference, and for CoreML to determine
#input and output shapes.

def refine_detections_graph(rois,
                            classifications,
                            window,
                            BBOX_STD_DEV,
                            DETECTION_MIN_CONFIDENCE,
                            DETECTION_MAX_INSTANCES,
                            DETECTION_NMS_THRESHOLD):
    """Refine classified proposals and filter overlaps and return final
    detections.

    Inputs:
        rois: [N, (y1, x1, y2, x2)] in normalized coordinates
        probs: [N, num_classes]. Class probabilities.
        deltas: [N, num_classes, (dy, dx, log(dh), log(dw))]. Class-specific
                bounding box deltas.
        window: (y1, x1, y2, x2) in normalized coordinates. The part of the image
            that contains the image excluding the padding.

    Returns detections shaped: [num_detections, (y1, x1, y2, x2, class_id, score)] where
        coordinates are normalized.
    """
    # Class IDs per ROI
    class_ids = tf.to_int32(classifications[:,4])
    # Class-specific bounding box deltas
    deltas_specific = classifications[:,0:4]
    # Class probability of the top class of each ROI
    class_scores = classifications[:,5]
    # Apply bounding box deltas
    # Shape: [boxes, (y1, x1, y2, x2)] in normalized coordinates
    refined_rois = apply_box_deltas_graph(
        rois, deltas_specific * BBOX_STD_DEV)
    # Clip boxes to image window
    refined_rois = clip_boxes_graph(refined_rois, window)

    # TODO: Filter out boxes with zero area
    # Filter out background boxes
    keep = tf.where(class_ids > 0)[:, 0]
    # Filter out low confidence boxes
    if DETECTION_MIN_CONFIDENCE:
        conf_keep = tf.where(class_scores >= DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = to_dense(keep)[0]

    # Apply per-class NMS
    # 1. Prepare variables
    pre_nms_class_ids = tf.gather(class_ids, keep)
    pre_nms_scores = tf.gather(class_scores, keep)
    pre_nms_rois = tf.gather(refined_rois,   keep)
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]

    def nms_keep_map(class_id):
        """Apply Non-Maximum Suppression on ROIs of the given class."""
        # Indices of ROIs of the given class
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        # Apply NMS
        class_keep = tf.image.non_max_suppression(
                tf.gather(pre_nms_rois, ixs),
                tf.gather(pre_nms_scores, ixs),
                max_output_size=DETECTION_MAX_INSTANCES,
                iou_threshold=DETECTION_NMS_THRESHOLD)
        # Map indices
        class_keep = tf.gather(keep, tf.gather(ixs, class_keep))
        # Pad with -1 so returned tensors have the same shape
        gap = DETECTION_MAX_INSTANCES - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)],
                            mode='CONSTANT', constant_values=-1)
        # Set shape so map_fn() can infer result shape
        class_keep.set_shape([DETECTION_MAX_INSTANCES])
        return class_keep

    # 2. Map over class IDs
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids,dtype=tf.int64, parallel_iterations=1)
    # 3. Merge results into one list, and remove -1 padding
    nms_keep = tf.reshape(nms_keep, [-1])
    nms_keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])
    # 4. Compute intersection between keep and nms_keep
    keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                    tf.expand_dims(nms_keep, 0))
    keep = to_dense(keep)[0]
    # Keep top detections
    roi_count = DETECTION_MAX_INSTANCES
    class_scores_keep = tf.gather(class_scores, keep)
    num_keep = tf.minimum(tf.shape(class_scores_keep)[0], roi_count)
    top_ids = tf.nn.top_k(class_scores_keep, k=num_keep, sorted=True)[1]
    keep = tf.gather(keep, top_ids)

    # Arrange output as [N, (y1, x1, y2, x2, class_id, score)]
    # Coordinates are normalized.
    detections = tf.concat([
        tf.gather(refined_rois, keep),
        tf.to_float(tf.gather(class_ids, keep))[..., tf.newaxis],
        tf.gather(class_scores, keep)[..., tf.newaxis]
        ], axis=1)

    # Pad with zeros if detections < DETECTION_MAX_INSTANCES
    gap = DETECTION_MAX_INSTANCES - tf.shape(detections)[0]
    detections = tf.pad(detections, [(0, gap), (0, 0)], "CONSTANT")
    return tf.reshape(detections, (-1,6))


class DetectionLayer(Layer):
    """Takes classified proposal boxes and their bounding box deltas and
    returns the final detection boxes.

    Returns:
    [batch, num_detections, (y1, x1, y2, x2, class_id, class_score)] where
    coordinates are normalized.
    """

    def __init__(self,
                 max_detections,
                 bounding_box_std_dev,
                 detection_min_confidence,
                 detection_nms_threshold,
                 image_shape,
                 **kwargs):
        super(DetectionLayer, self).__init__(**kwargs)
        self.max_detections = max_detections
        self.bounding_box_std_dev = bounding_box_std_dev
        self.detection_min_confidence = detection_min_confidence
        self.detection_nms_threshold = detection_nms_threshold
        self.image_shape = image_shape
        assert max_detections != None

    def get_config(self):
        config = super(DetectionLayer, self).get_config()
        config['max_detections'] = self.max_detections
        config['bounding_box_std_dev'] = self.bounding_box_std_dev
        config['detection_min_confidence'] = self.detection_min_confidence
        config['detection_nms_threshold'] = self.detection_nms_threshold
        config['image_shape'] = self.image_shape
        return config

    def call(self, inputs):
        def refine_detections(inputs):
            rois = inputs[0]
            classifications = inputs[1]

            if len(inputs) > 2:
                image_bounding_box = inputs[2]
            else:
                image_bounding_box = tf.convert_to_tensor(np.array([0, 0, 1, 1]),
                                                          dtype=tf.float32)

            return refine_detections_graph(rois,
                                           classifications,
                                           image_bounding_box,
                                           np.array(self.bounding_box_std_dev),
                                           self.detection_min_confidence,
                                           self.max_detections,
                                           self.detection_nms_threshold)

        detections = keras.layers.Lambda(lambda x: tf.map_fn(refine_detections, x, dtype=tf.float32))(inputs)
        return detections

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        roi_shape, classifications_shape = input_shape
        return (roi_shape[0],self.max_detections, 6)
