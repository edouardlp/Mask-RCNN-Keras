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

    return np.array([width,
                     height,
                     model_input_shape[2]],
                    dtype=np.int32),\
           np.array(padding,
                    dtype=np.int32)

def denormalize_box(box,
                    original_shape,
                    model_input_shape,
                    mode=NormalizationModeKeys.ASPECT_FIT_PAD):

    width_ratio = model_input_shape[0] / original_shape[0]
    height_ratio = model_input_shape[1] / original_shape[1]

    fit_to_width = width_ratio < height_ratio

    short_side = original_shape[1] if fit_to_width else original_shape[0]
    short_side_input_in_original = model_input_shape[1] / width_ratio if fit_to_width else model_input_shape[0] / height_ratio
    margins = short_side_input_in_original - short_side

    denorm_width_factor = original_shape[0] if fit_to_width else short_side_input_in_original
    denorm_height_factor = short_side_input_in_original if fit_to_width else original_shape[1]

    if(mode == NormalizationModeKeys.FILL):
        margins = 0
        denorm_width_factor = original_shape[0]
        denorm_height_factor = original_shape[1]

    y1 = box[0]
    x1 = box[1]
    y2 = box[2]
    x2 = box[3]
    width = x2 - x1
    height = y2 - y1

    x1 = x1 * denorm_width_factor
    x1 = x1 if fit_to_width else x1 - margins / 2
    y1 = y1 * denorm_height_factor
    y1 = y1 - margins / 2 if fit_to_width else y1
    width = width * denorm_width_factor
    height = height * denorm_height_factor
    return [x1, y1, width, height]