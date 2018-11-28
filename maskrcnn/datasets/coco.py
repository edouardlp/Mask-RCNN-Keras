import tensorflow as tf
def make_dataset_fn(image_width,
                    image_height,
                    image_channels,
                    batch_size):

    def make_dataset(path,
                      epochs=1):

        dataset = tf.data.TFRecordDataset(path, compression_type="GZIP")

        def parser(record):
            IMAGE_KEY = "image"
            keys_to_features = {
                IMAGE_KEY: tf.FixedLenFeature((), tf.string),
            }
            parsed = tf.parse_single_example(record, keys_to_features)
            image = tf.decode_raw(parsed[IMAGE_KEY], tf.float32)
            image = tf.reshape(image, [image_width, image_height, image_channels])
            print(image.shape)
            return image

        dataset = dataset.map(parser)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(epochs)
        return  dataset
    return make_dataset

dataset_fn = make_dataset_fn(1024,1024,3,2)

class COCODataset():

    train_preprocessed = False
    eval_preprocessed = False

    def __init__(self):
        pass

    def preprocess_train(self):
        pass

    def preprocess_eval(self):
        pass

    def make_train_input_fn(self):
        pass

    def make_eval_input_fn(self):
        def eval_input_fn():
            return dataset_fn("Data/coco/data.tfrecords")
        return eval_input_fn