"""Module for testing loss."""


import tensorflow as tf
import numpy as np
import pytest

from coral_ordinal import loss as coral_loss
from coral_ordinal import layer


def _create_test_data():
    np.random.seed(10)

    X = np.random.normal(size=(8, 99))
    y = np.array([0, 1, 2, 2, 2, 3, 4, 4])
    return X, y


def test_corn_loss():
    X, y = _create_test_data()
    corn_loss = coral_loss.CornOrdinalCrossEntropy()
    num_classes = len(np.unique(y))
    corn_net = layer.CornOrdinal(num_classes=num_classes, input_dim=X.shape[1])
    logits = corn_net(X)
    assert logits.shape == (8, num_classes - 1)

    loss = corn_loss(y_true=y, y_pred=logits)
    assert loss.numpy() == pytest.approx(3.76, 0.1)
