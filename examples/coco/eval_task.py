from maskrcnn.model import MaskRCNNModel
from maskrcnn.datasets.coco import COCODataset

model = MaskRCNNModel("", initial_keras_weights="Data/weights.h5")

coco_dataset = COCODataset()
results = model.predict(coco_dataset)
for result in results:
    print(result)