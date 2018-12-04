import os
import tensorflow as tf
import PIL
import numpy as np

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from ..utils import NormalizationModeKeys
from ..utils import normalized_image_shape_and_padding
from ..utils import denormalize_box
from ..utils import crop_box_to_outer_box

class COCODataset():

    _coco = None

    _ID_KEY = "id"
    _IMAGE_KEY = "image"
    _ORIGINAL_SHAPE_KEY = "original_shape"
    _ACTUAL_SHAPE_KEY = "actual_shape"
    _IMAGE_PADDING_KEY = "image_padding"
    _IMAGE_BOUNDING_BOX_KEY = "image_bounding_box"

    _CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

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
        self.mode = NormalizationModeKeys.ASPECT_FIT_PAD

    def preprocess(self, reprocess_if_exists=True, limit=None):
        self._preprocess(type=self.type,
                         output_path=self.path,
                         reprocess_if_exists=reprocess_if_exists,
                         limit=limit)

    def make_input_fn(self, batch_size, epochs=1, shuffle=True, limit=None):
        self.preprocess(reprocess_if_exists=False)
        return lambda : self._make_dataset(path=self.path,
                                           batch_size=batch_size,
                                           epochs=epochs,
                                           limit=limit)

    def evaluate_results(self, results):
        coco = self._get_coco()

        cat_ids = coco.getCatIds()
        categories = coco.loadCats(cat_ids)

        mapping = []
        supported_cat_ids = []

        for class_name in self._CLASS_NAMES:
            id_found = None
            for category in categories:
                if category["name"] == class_name:
                    id_found = category["id"]
                    supported_cat_ids.append(id_found)
                    break
            if id_found:
                mapping.append(id_found)
            else:
                mapping.append(0)

        coco_results = []
        image_ids = []

        for result in results:
            id = int(result.image_info.id)
            image_bounding_box = result.image_info.bounding_box
            image_original_shape = result.image_info.original_shape
            image_ids.append(id)
            for detection in result.detections:
                cropped_box = crop_box_to_outer_box(detection.bounding_box, image_bounding_box)
                bounding_box = denormalize_box(cropped_box,
                                               image_original_shape)
                coco_result = {"image_id": id,
                               "category_id" : mapping[detection.class_id],
                               "score" : detection.probability,
                               "bbox" : bounding_box}
                coco_results.append(coco_result)

        coco_results = coco.loadRes(coco_results)
        cocoEval = COCOeval(coco, coco_results, "bbox")
        cocoEval.params.imgIds = image_ids
        cocoEval.params.catIds = supported_cat_ids
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

    def class_label_from_id(self, id):
        return self._CLASS_NAMES[int(id)]

    def _get_coco(self):
        if self._coco is None:
            instance_file = '{}/instances_{}{}.json'.format(self.annotations_dir,
                                                            self.type,
                                                            self.year)
            self._coco = COCO(instance_file)
        return self._coco

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
        coco = self._get_coco()

        imgIds = coco.getImgIds()
        imgIds = sorted(imgIds)

        if limit:
            imgIds = imgIds[:limit]

        os.makedirs(os.path.dirname(output_path), exist_ok=reprocess_if_exists)

        if not os.path.isdir(self.images_dir):
            os.makedirs(self.images_dir, exist_ok=False)
            coco.download(tarDir=self.images_dir, imgIds=imgIds)

        options = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
        writer = tf.python_io.TFRecordWriter(output_path, options=options)

        images = coco.loadImgs(imgIds)

        for image in images:
            example = self._make_example(image)
            writer.write(example)
        writer.close()

    def _make_dataset(self,
                      path,
                      batch_size,
                      epochs=1,
                      limit=None):
        dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")
        dataset = dataset.map(self._make_record_example_parser_fn())
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epochs)
        if limit:
            return dataset.take(limit)
        return dataset

    def _make_example(self, image):
        filename = image["file_name"]

        original = load_img(self.images_dir + "/" + filename)
        original_shape = np.array([image["width"],image["height"],3], dtype=np.int64)
        normalized_shape, normalized_padding = normalized_image_shape_and_padding(original_shape, self.image_shape, mode=self.mode)
        resize_shape = normalized_shape[0:2]
        resized_image = original.resize(resize_shape, resample=PIL.Image.NEAREST)
        resized_image_array = img_to_array(resized_image)

        actual_shape = np.array(resized_image_array.shape)

        def shape_to_feature(shape):
            return tf.train.Feature(int64_list=tf.train.Int64List(value=shape.astype(np.int64)))

        def string_to_feature(string):
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(string)]))

        def image_to_feature(image):
            return string_to_feature(image.flatten().tostring())

        feature = {self._ID_KEY: string_to_feature(str(image['id'])),
                   self._ORIGINAL_SHAPE_KEY: shape_to_feature(original_shape),
                   self._ACTUAL_SHAPE_KEY: shape_to_feature(actual_shape),
                   self._IMAGE_PADDING_KEY : shape_to_feature(normalized_padding.flatten()),
                   self._IMAGE_KEY: image_to_feature(resized_image_array)}
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        return example.SerializeToString()

    def _make_record_example_parser_fn(self):
        def parser(record):
            keys_to_features = {
                self._ID_KEY: tf.FixedLenFeature([], tf.string),
                self._IMAGE_KEY: tf.FixedLenFeature([], tf.string),
                self._IMAGE_PADDING_KEY: tf.FixedLenFeature([6], tf.int64),
                self._ORIGINAL_SHAPE_KEY: tf.FixedLenFeature([3], tf.int64),
                self._ACTUAL_SHAPE_KEY: tf.FixedLenFeature([3], tf.int64),
            }
            features = tf.parse_single_example(record, keys_to_features)

            id = features[self._ID_KEY]
            image = tf.decode_raw(features[self._IMAGE_KEY], tf.float32)
            original_shape = tf.cast(features[self._ORIGINAL_SHAPE_KEY], tf.int32)
            actual_shape = tf.cast(features[self._ACTUAL_SHAPE_KEY], tf.int32)
            padding = tf.cast(features[self._IMAGE_PADDING_KEY], tf.int32)
            padding = tf.reshape(padding, shape=(3,2))

            image = tf.reshape(image, actual_shape)
            padded_image = tf.pad(image, padding, "CONSTANT")

            #TODO: permute self.image_shape so it is HWC
            padded_image = tf.reshape(padded_image, self.image_shape)

            image_shape = tf.convert_to_tensor(self.image_shape, dtype=tf.float32)

            padding_float = tf.to_float(padding)
            image_shape_float = tf.to_float(image_shape)
            actual_shape_float = tf.to_float(actual_shape)

            #actual_shape is store HWC
            actual_height = actual_shape_float[0]
            actual_width = actual_shape_float[1]

            y1 = padding_float[0][0]
            x1 = padding_float[1][0]
            y2 = y1+actual_height
            x2 = x1+actual_width

            bounding_box = tf.convert_to_tensor([y1/image_shape_float[1],
                                                 x1/image_shape_float[0],
                                                 y2/image_shape_float[1],
                                                 x2/image_shape_float[0]], dtype=tf.float32)

            prefix = "input_"
            return {prefix+self._ID_KEY : id,
                    prefix+self._IMAGE_KEY : padded_image,
                    prefix+self._IMAGE_BOUNDING_BOX_KEY : bounding_box,
                    prefix+self._ORIGINAL_SHAPE_KEY: original_shape}

        return parser