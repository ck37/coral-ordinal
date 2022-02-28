"""Module for testing loss."""


import tensorflow as tf
import numpy as np
import pytest

from coral_ordinal import loss as coral_loss
from coral_ordinal import layer


def _create_test_data():
    # Test data from example in
    # https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    np.random.seed(10)

    X = np.random.normal(size=(8, 99))
    y = np.array([0, 1, 2, 2, 2, 3, 4, 4])
    return X, y


def test_corn_loss():
    X, y = _create_test_data()
    corn_loss = coral_loss.CornOrdinalCrossEntropy()
    num_classes = len(np.unique(y))
    tf.random.set_seed(1)
    corn_net = layer.CornOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = corn_net(X)
    assert logits.shape == (8, num_classes - 1)

    loss = corn_loss(y_true=y, y_pred=logits)
    # see https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    # for approximately same value for pytorch immplementation.
    assert loss.numpy() == pytest.approx(3.54, 0.01)
