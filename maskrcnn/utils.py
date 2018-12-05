import PIL
import numpy as np
from math import floor

class NormalizationModeKeys(object):
  ASPECT_FIT_PAD = 'aspect_fit_pad'
  FILL = 'fill'

def normalized_image_shape_and_padding(original_shape,
                                       model_input_shape,
                                       mode = NormalizationModeKeys.ASPECT_FIT_PAD):

    if(mode == NormalizationModeKeys.FILL):
        return model_input_shape, np.zeros((3,2),dtype=np.int32)

    width_multiplier = model_input_shape[0] / original_shape[0]
    height_multiplier = model_input_shape[1] / original_shape[1]

    fit_to_width = width_multiplier < height_multiplier

    actual_multiplier = width_multiplier if fit_to_width else height_multiplier

    original_short_side = original_shape[1] if fit_to_width else original_shape[0]

    original_short_side_in_input =  floor(original_short_side * actual_multiplier)
    input_short_side = model_input_shape[1] if fit_to_width else model_input_shape[0]

    margins = input_short_side - original_short_side_in_input

    width = floor(original_shape[0]*actual_multiplier)
    height = floor(original_shape[1]*actual_multiplier)

    padding_width = floor(margins/2)

    horizontal_padding = 0 if fit_to_width else padding_width
    horizontal_padding_rounding_error = 0 if fit_to_width else input_short_side-(padding_width*2)-width
    vertical_padding = padding_width if fit_to_width else 0
    vertical_padding_rounding_error = input_short_side-(padding_width*2)-height if fit_to_width else 0

    left_padding = horizontal_padding
    right_padding = horizontal_padding+horizontal_padding_rounding_error
    top_padding = vertical_padding
    bottom_padding = vertical_padding+vertical_padding_rounding_error

    padding = [[top_padding,
               bottom_padding],
               [left_padding,
               right_padding],
               [0,0]]

    return np.array([height,
                     width,
                     model_input_shape[2]],
                    dtype=np.int32),\
           np.array(padding,
                    dtype=np.int32)

#This accepts and outputs normalized coordinates
def crop_box_to_outer_box_normed(box, cropping_box):

    crop_y1 = cropping_box[0]
    crop_x1 = cropping_box[1]
    crop_y2 = cropping_box[2]
    crop_x2 = cropping_box[3]

    crop_width = crop_x2-crop_x1
    crop_height = crop_y2-crop_y1

    y1 = max(box[0]-crop_y1,0)
    x1 = max(box[1]-crop_x1,0)
    y2 = min(box[2]-crop_y1,crop_height)
    x2 = min(box[3]-crop_x1,crop_width)
    #We renormalize the box to the shape of the crop
    return [y1/crop_height, x1/crop_width, y2/crop_height, x2/crop_width]

def crop_box_to_outer_box(box, cropping_box):

    crop_x1 = cropping_box[0]
    crop_y1 = cropping_box[1]
    crop_width = cropping_box[2]
    crop_height = cropping_box[3]

    x1 = max(box[0]-crop_x1,0)
    y1 = max(box[1]-crop_y1,0)
    width = min(box[2], crop_width)
    height = min(box[3], crop_height)

    return [x1,
            y1,
            width,
            height]

def denormalize_box(box,
                    shape):
    y1 = box[0]
    x1 = box[1]
    y2 = box[2]
    x2 = box[3]
    width = x2 - x1
    height = y2 - y1
    return scale_box([x1, y1, width, height], shape[0], shape[1])

def scale_box(box, horizontal_scale, vertical_scale):

    y1 = box[0]
    x1 = box[1]
    width = box[2]
    height = box[3]

    x1 = x1 * horizontal_scale
    y1 = y1 * vertical_scale
    width = width * horizontal_scale
    height = height * vertical_scale
    return [x1, y1, width, height]
