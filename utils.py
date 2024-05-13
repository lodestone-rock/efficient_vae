from flax import struct
from typing import Callable
import numpy as np
import cv2

def get_overlapping_keys(dict_a, dict_b):
    return [key for key in dict_a.keys() if key in dict_b]

def exclude_keys_from_list(keys_list, exclude_keys):
    return [key for key in keys_list if key not in exclude_keys]

def flatten_dict(dictionary, parent_key='', sep='.'):
    items = []
    for key, value in dictionary.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep=sep).items())
        else:
            items.append((new_key, value))
    return dict(items)

def unflatten_dict(dictionary, sep='.'):
    unflattened_dict = {}
    for key, value in dictionary.items():
        keys = key.split(sep)
        current_dict = unflattened_dict
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    return unflattened_dict

def create_image_mosaic(images, rows, cols, output_file):
    n, h, w, c = images.shape
    mosaic = np.zeros((h * rows, w * cols, c), dtype=np.uint8)
    
    for i in range(min(n, rows * cols)):
        row = i // cols
        col = i % cols
        mosaic[row*h:(row+1)*h, col*w:(col+1)*w, :] = images[i]
    
    cv2.imwrite(output_file, mosaic)

class FrozenModel(struct.PyTreeNode):
    """
    mimic the behaviour of train_state but this time for frozen params
    to make it passable to the jitted function
    """

    # use pytree_node=False to indicate an attribute should not be touched
    # by Jax transformations.
    call: Callable = struct.field(pytree_node=False)
    params: dict = struct.field(pytree_node=True)
