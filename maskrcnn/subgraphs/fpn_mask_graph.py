import tensorflow as tf
if tf.__version__ == '1.5.0':
    import keras
else:
    from tensorflow import keras
import numpy as np

from .pyramid_roi_align_layer import PyramidROIAlign

#We use a custom layer to implement the time distributed layer in Swift
class TimeDistributedMask(keras.layers.Layer):

    def __init__(self,
                 max_regions,
                 pool_size,
                 **kwargs):
        super(TimeDistributedMask, self).__init__(**kwargs)
        self.max_regions = max_regions
        self.pool_size = pool_size

    def get_config(self):
        config = super(TimeDistributedMask, self).get_config()
        config['max_regions'] = self.max_regions
        config['pool_size'] = self.pool_size
        return config

    def call(self, inputs):
        return tf.zeros((self.pool_size*2,self.pool_size*2,self.max_regions))

    def compute_output_shape(self, input_shape):
        return (None,self.pool_size*2, self.pool_size*2, self.max_regions)


class FPNMaskGraph():

    def __init__(self,
                 rois,
                 feature_maps,
                 pool_size,
                 image_shape,
                 num_classes,
                 max_regions,
                 pyramid_top_down_size):

        self.rois = rois
        self.feature_maps = feature_maps
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.max_regions = max_regions
        self.pyramid_top_down_size = pyramid_top_down_size
        assert max_regions != None

    def _build_coreml_inner_model(self):

        num_classes = self.num_classes
        pyramid_top_down_size = self.pyramid_top_down_size

        input = keras.layers.Input((self.pool_size, self.pool_size, pyramid_top_down_size))

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv1")(input)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn1')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv2")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn2')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv3")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn3')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2D(256, (3, 3), padding="same", name="mrcnn_mask_conv4")(x)
        x = keras.layers.BatchNormalization(name='mrcnn_mask_bn4')(x, training=False)
        x = keras.layers.Activation('relu')(x)

        x = keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu", name="mrcnn_mask_deconv")(x)
        #TODO: try to only perform the convolution for the relevant class
        x = keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid", name="mrcnn_mask")(x)
        #TODO: select the output accoding to the class
        x = keras.layers.Reshape((self.pool_size * 2, self.pool_size * 2, num_classes))(x)

        return keras.models.Model(inputs=[input], outputs=[x])

    def build(self, environment):

        rois = self.rois
        feature_maps = self.feature_maps

        pool_size = self.pool_size
        image_shape = self.image_shape
        num_classes = self.num_classes
        max_regions = self.max_regions
        pyramid_top_down_size = self.pyramid_top_down_size

        pyramid = PyramidROIAlign(name="roi_align_mask",
                                  pool_shape=[pool_size, pool_size],
                                  image_shape=image_shape)([rois] + feature_maps)

        if environment == "coreml":
            result = TimeDistributedMask(max_regions=max_regions,
                                         pool_size=pool_size,
                                         name="masks")([pyramid, rois])
            fpn_mask_model = self._build_coreml_inner_model()
            return fpn_mask_model, result
        else:
            x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(
                pyramid)
            x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(), name='mrcnn_mask_bn1')(x,
                                                                                                       training=False)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv2")(
                x)
            x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(), name='mrcnn_mask_bn2')(x,
                                                                                                       training=False)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv3")(
                x)
            x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(), name='mrcnn_mask_bn3')(x,
                                                                                                       training=False)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv4")(
                x)
            x = keras.layers.TimeDistributed(keras.layers.BatchNormalization(), name='mrcnn_mask_bn4')(x,
                                                                                                       training=False)
            x = keras.layers.Activation('relu')(x)

            x = keras.layers.TimeDistributed(keras.layers.Conv2DTranspose(256, (2, 2), strides=2, activation="relu"),
                                             name="mrcnn_mask_deconv")(x)
            x = keras.layers.TimeDistributed(keras.layers.Conv2D(num_classes, (1, 1), strides=1, activation="sigmoid"),
                                             name="mrcnn_mask")(x)
            masks = keras.layers.Reshape((self.max_regions, self.pool_size * 2, self.pool_size * 2, num_classes))(x)

            def extract_mask(inputs):
                roi = inputs[0]
                roi_masks = inputs[1]
                class_id = roi[4]
                class_id = tf.to_int32(class_id)
                return roi_masks[:, :, class_id]

            def extract_relevant_class(inputs):
                result = tf.map_fn(extract_mask, inputs, dtype=tf.float32)
                #TODO: not sure this transpose is required
                #result = tf.transpose(result, perm=[0, 2, 1])
                return result

            result = keras.layers.Lambda(lambda x: tf.map_fn(extract_relevant_class,x, dtype=tf.float32), name="masks")([rois,masks])
            return None,result