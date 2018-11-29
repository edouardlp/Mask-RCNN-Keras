import numpy as np

class MaskRCNNImageInfo():

    def __init__(self,
                 id,
                 original_shape):

        self.id = id
        self.original_shape = original_shape

class MaskRCNNResult():

    def __init__(self,
                 dataset_id,
                 image_info,
                 detections):

        self.dataset_id = dataset_id
        self.image_info = image_info
        self.detections = detections

class MaskRCNNDetectionResult():

    def __init__(self,
                 probability,
                 class_id,
                 class_label,
                 bounding_box,
                 mask):

        self.probability = probability
        self.class_id = class_id
        self.class_label = class_label
        self.bounding_box = bounding_box
        self.mask = mask


def _results_from_tensor_values(values,
                                dataset_id,
                                image_info_fn,
                                class_label_fn):

    results = []
    for index,value in enumerate(values):

        image_info = image_info_fn(index)
        raw_detections = value["detections"]
        masks = value["masks"]
        masks = np.transpose(masks, axes=[2, 1, 0])

        detections = []

        for i in range(0, raw_detections.shape[0]):
            raw_detection = raw_detections[i]
            mask = masks[i]
            class_id = int(raw_detection[4])
            if (class_id == 0):
                continue
            probability = float(raw_detection[5])
            bounding_box = raw_detection[0:4].tolist()
            detection = MaskRCNNDetectionResult(probability=probability,
                                                class_id=class_id,
                                                class_label=class_label_fn(class_id),
                                                bounding_box=bounding_box,
                                                mask=mask)
            detections.append(detection)

        result = MaskRCNNResult(dataset_id=dataset_id,
                                    image_info=image_info,
                                    detections=detections)
        results.append(result)
        return results