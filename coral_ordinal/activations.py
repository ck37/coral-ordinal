"""Functions to convert logits to probabilities (CDF) and softmax.

Also conversion from probabilities to labels.
"""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def coral_cumprobs(logits: tf.Tensor) -> tf.Tensor:
    """Turns logits from CORAL layer into cumulative probabilities."""
    return tf.math.sigmoid(logits)


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def corn_cumprobs(logits: tf.Tensor, axis=-1) -> tf.Tensor:
    """Turns logits from CORN layer into cumulative probabilities."""
    probs = tf.math.sigmoid(logits)
    return tf.math.cumprod(probs, axis=1)


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def cumprobs_to_softmax(cumprobs: tf.Tensor) -> tf.Tensor:
    """Turns ordinal probabilities into label probabilities (softmax)."""

    # Number of columns is the number of classes - 1
    num_classes = cumprobs.shape[1] + 1

    # Create a list of tensors.
    probs = []

    # First, get probability predictions out of the cumulative logits.
    # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
    # Pr(Y = 0) = 1 - s(logit for column 0)
    probs.append(1.0 - cumprobs[:, 0])

    # For the other columns, the probability is:
    # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
    if num_classes > 2:
        for val in range(1, num_classes - 1):
            probs.append(cumprobs[:, val - 1] - cumprobs[:, val])

    # Special handling of the maximum label value.
    probs.append(cumprobs[:, num_classes - 2])

    # Combine as columns into a new tensor.
    probs_tensor = tf.concat(tf.transpose(probs), axis=1)

    return probs_tensor


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def cumprobs_to_label(cumprobs: tf.Tensor, threshold: float = 0.5) -> tf.Tensor:
    """Converts cumulative probabilities for ordinal data to a class label.

    Converts probabilities of the form

        [Pr(y > 0), Pr(y > 1), ..., Pr(y > K-1)]

    to a predicted label as one of [0, ..., K-1].

    By default, it uses the natural threshold of 0.5 to pick the label.
    Can be changed to be more/less conservative.

    Args:
      cumprobs: tensor with cumulative probabilities from 0..K-1.
      threshold: which threshold to choose for the label prediction.
        Defaults to the natural threshold of 0.5.

    Returns:
      A tensor of one column, with the label (integer).
    """
    assert 0 < threshold < 1, f"threshold must be in (0, 1). Got {threshold}."
    predict_levels = tf.cast(cumprobs > threshold, dtype=tf.int32)
    predicted_labels = tf.reduce_sum(predict_levels, axis=1)
    return predicted_labels


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def ordinal_softmax(x, axis=-1):
    """Convert the ordinal logit output of CoralOrdinal() to label probabilities.

    Args:
      x: Logit output of the CoralOrdinal() layer.
      axis: Not yet supported.
    """
    # Convert the ordinal logits into cumulative probabilities.
    cum_probs = coral_cumprobs(x)
    return cumprobs_to_softmax(cum_probs)


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def corn_ordinal_softmax(logits: tf.Tensor) -> tf.Tensor:
    """Turns CORN logits into label probabilities."""
    cum_probs = corn_cumprobs(logits)
    return cumprobs_to_softmax(cum_probs)
