import os
import tensorflow as tf
from pycocotools.coco import COCO

from maskrcnn.results import MaskRCNNImageInfo

class COCODataset():

    def __init__(self,
                 path,
                 type,
                 year,
                 image_shape,
                 annotations_dir=None,
                 images_dir=None):

        self.id = "coco-"+year
        assert type in ['train', 'val']
        self.type = type
        self.path = path
        self.year = year
        self.image_shape = image_shape
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir

    def preprocess(self, reprocess_if_exists=True, limit=None):
        self._preprocess(type=self.type,
                         output_path=self.path,
                         reprocess_if_exists=reprocess_if_exists,
                         limit=limit)

    def make_train_input_fn(self, batch_size, epochs=1, shuffle=True, limit=None):

        self.preprocess(reprocess_if_exists=False)

        return lambda : self._make_dataset(path=self.path,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           shuffle=shuffle,
                                           limit=limit)

    def make_eval_input_fn(self, batch_size, shuffle=False, limit=None):

        self.preprocess(reprocess_if_exists=False)

        return lambda: self._make_dataset(path=self.path,
                                          batch_size=batch_size,
                                          epochs=1,
                                          shuffle=shuffle,
                                          limit=limit)

    def make_predict_input_and_info_fns(self, batch_size, limit=None):

        self.preprocess(reprocess_if_exists=False)

        input_fn = lambda: self._make_dataset(path=self.path,
                                          batch_size=batch_size,
                                          epochs=1,
                                          shuffle=False,
                                          limit=limit)

        def info_fn(index):
            #TODO: return MaskRCNNImageInfo
            return index

        return input_fn, info_fn

    def evaluate_results(self, results):
        pass

    def class_label_from_id(self, id):
        #TODO:
        return "label"

    def _preprocess(self,
                    type,
                    output_path,
                    reprocess_if_exists,
                    limit):

        #TODO: check if exists
        exists = True
        reprocess = reprocess_if_exists or (not exists)

        if not reprocess:
            return

        #TODO: support no annotations_dir (download if possible)

        instance_file = '{}/instances_{}_{}.json'.format(self.annotations_dir, type,self.year)
        coco = COCO(instance_file)
        imgIds = coco.getImgIds()
        if limit:
            imgIds = imgIds[:limit]
        imgs = coco.loadImgs(imgIds)

        os.makedirs(os.path.dirname(output_path), exist_ok=reprocess_if_exists)

        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir, exist_ok=False)
            coco.download(tarDir=self.images_dir, imgIds=imgIds)

        for img in imgs:
            id = img['id']

    def _make_dataset(self,
                      path,
                      batch_size,
                      epochs=1,
                      shuffle=True,
                      limit=None):
        dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")

        def parser(record):
            IMAGE_KEY = "image"
            keys_to_features = {
                IMAGE_KEY: tf.FixedLenFeature((), tf.string),
            }
            parsed = tf.parse_single_example(record, keys_to_features)
            image = tf.decode_raw(parsed[IMAGE_KEY], tf.float32)
            #TODO: scale up as needed from the shape specified in the tf example (if it does not equal self.image_shape)
            image = tf.reshape(image, self.image_shape)
            return image

        dataset = dataset.map(parser)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epochs)
        if shuffle:
            #TODO: increase the buffer size, but to what?
            dataset = dataset.shuffle(batch_size)
        if limit:
            print(limit)
            return dataset.take(limit)
        return dataset