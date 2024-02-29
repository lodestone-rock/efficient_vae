from flax import struct
from typing import Callable

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

class FrozenModel(struct.PyTreeNode):
    """
    mimic the behaviour of train_state but this time for frozen params
    to make it passable to the jitted function
    """

    # use pytree_node=False to indicate an attribute should not be touched
    # by Jax transformations.
    call: Callable = struct.field(pytree_node=False)
    params: dict = struct.field(pytree_node=True)