from maskrcnn.model import MaskRCNNModel
from maskrcnn.datasets.coco import COCODataset

model = MaskRCNNModel("", initial_keras_weights="Data/weights.h5")

coco_dataset = COCODataset(path="Data/coco/data.tfrecords",
                           type='val',
                           year='2017',
                           image_shape=(1024,1024,3))

input_fn, image_info_fn = coco_dataset.make_predict_input_and_info_fns(batch_size=1, limit=1)

results = model.predict(dataset_id=coco_dataset.id,
                        input_fn=input_fn,
                        image_info_fn=image_info_fn,
                        class_label_fn=coco_dataset.class_label_from_id)

coco_dataset.evaluate_results(results)