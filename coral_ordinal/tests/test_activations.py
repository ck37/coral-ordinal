"""Tests activations."""

import numpy as np

from coral_ordinal import activations


def test_ordinal_probs_to_label():
    probas = np.array(
        [
            [0.934, 0.861, 0.323, 0.492, 0.295],
            [0.496, 0.485, 0.267, 0.124, 0.058],
            [0.985, 0.967, 0.920, 0.819, 0.506],
        ]
    )
    labels = activations.cumprobs_to_label(probas).numpy()
    np.testing.assert_allclose(labels, np.array([2, 0, 5]))
