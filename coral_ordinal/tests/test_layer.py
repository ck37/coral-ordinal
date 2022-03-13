import tensorflow as tf
import numpy as np
import pytest

from coral_ordinal import layer


def _create_test_data():
    # Test data from example in
    # https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    np.random.seed(10)

    X = np.random.normal(size=(8, 99))
    y = np.array([0, 1, 2, 2, 2, 3, 4, 4])
    return X, y


def test_corn_layer():
    corn_layer = layer.CornOrdinal(num_classes=4, kernel_initializer="uniform")
    corn_layer_config = corn_layer.get_config()

    corn_layer2 = layer.CornOrdinal(**corn_layer_config)

    assert isinstance(corn_layer2, layer.CornOrdinal)
