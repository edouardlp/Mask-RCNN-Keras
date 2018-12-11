import numpy as np

from maskrcnn import results_pb2

def _results_from_tensor_values(values,
                                dataset_id,
                                class_label_fn):

    results = results_pb2.Results()

    for index,value in enumerate(values):

        id = value["input_id"].decode('utf-8')

        result = results.results.add()
        result.imageInfo.datasetId = dataset_id
        result.imageInfo.id = id
        original_shape = value["input_original_shape"]
        result.imageInfo.width = int(original_shape[0])
        result.imageInfo.height = int(original_shape[1])

        raw_detections = np.copy(value["detections"])
        for i in range(0, raw_detections.shape[0]):
            raw_detection = raw_detections[i]
            # mask = masks[i]
            class_id = int(raw_detection[4])
            if (class_id == 0):
                continue
            probability = float(raw_detection[5])
            y1 = raw_detection[0]
            x1 = raw_detection[1]
            y2 = raw_detection[2]
            x2 = raw_detection[3]

            detection = result.detections.add()
            detection.probability = probability
            detection.classId = class_id
            detection.classLabel = class_label_fn(class_id)
            detection.boundingBox.origin.x = x1
            detection.boundingBox.origin.y = y1
            detection.boundingBox.size.width = y2-y1
            detection.boundingBox.size.height = x2-x1

    return results