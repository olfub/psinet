import numpy as np


def get_one_hot(targets, nb_classes=None):
    # source: https://stackoverflow.com/a/42874726
    if nb_classes is None:
        nb_classes = int(np.max(targets)) + 1

    flat_targets = np.array(targets).reshape(-1)
    res = np.eye(nb_classes, dtype=targets.dtype)[flat_targets]
    return res.reshape(list(flat_targets.shape) + [nb_classes])
