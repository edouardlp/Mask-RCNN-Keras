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
                 num_classes,
                 pyramid_top_down_size,
                 **kwargs):
        super(TimeDistributedMask, self).__init__(**kwargs)
        self.max_regions = max_regions
        self.pool_size = pool_size
        self.num_classes = num_classes
        self.pyramid_top_down_size = pyramid_top_down_size

    def get_config(self):
        config = super(TimeDistributedMask, self).get_config()
        config['max_regions'] = self.max_regions
        config['pool_size'] = self.pool_size
        config['num_classes'] = self.num_classes
        config['pyramid_top_down_size'] = self.pyramid_top_down_size
        return config

    #This will not get exported to CoreML
    def _build_keras_inner_graph(self):

        num_classes = self.num_classes

        def graph(input):
            x = keras.layers.TimeDistributed(keras.layers.Conv2D(256, (3, 3), padding="same"), name="mrcnn_mask_conv1")(
                input)
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
            x = keras.layers.Reshape((self.max_regions, self.pool_size * 2, self.pool_size * 2, num_classes))(x)
            return x

        return graph

    def call(self, inputs):
        pyramid = inputs[0]
        rois = inputs[1]
        graph = self._build_keras_inner_graph()
        masks = graph(pyramid)

        #We extract the masks corresponding to the class ids
        class_ids = rois[:,:,4]
        class_ids = tf.to_int32(class_ids)
        class_ids = tf.reshape(class_ids, [-1])
        slices = []
        for i in range(0,self.max_regions):
            m = masks[:, i, :, :, class_ids[i]]
            m= tf.expand_dims(m,axis=1)
            slices.append(m)

        #We convert back to channels last
        result = tf.concat(slices, axis=1)
        result = tf.transpose(result, perm=[0,3,2,1])
        return result

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

        result = TimeDistributedMask(max_regions=max_regions,
                                     pool_size=pool_size,
                                     num_classes=num_classes,
                                     pyramid_top_down_size=pyramid_top_down_size,
                                     name="masks")([pyramid, rois])
        fpn_mask_model = self._build_coreml_inner_model()
        return fpn_mask_model, result