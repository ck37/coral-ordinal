import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
def ordinal_softmax(x, axis=-1):
    """Convert the ordinal logit output of CoralOrdinal() to label probabilities.

    Args:
      x: Logit output of the CoralOrdinal() layer.
      axis: Not yet supported.
    """

    # Number of columns is the number of classes - 1
    num_classes = x.shape[1] + 1

    # Convert the ordinal logits into cumulative probabilities.
    cum_probs = tf.math.sigmoid(x)

    # Create a list of tensors.
    probs = []

    # First, get probability predictions out of the cumulative logits.
    # Column 0 is Probability that y > 0, so Pr(y = 0) = 1 - Pr(y > 0)
    # Pr(Y = 0) = 1 - s(logit for column 0)
    probs.append(1.0 - cum_probs[:, 0])

    # For the other columns, the probability is:
    # Pr(y = k) = Pr(y > k) - Pr(y > k - 1)
    if num_classes > 2:
        for val in range(1, num_classes - 1):
            probs.append(cum_probs[:, val - 1] - cum_probs[:, val])

    # Special handling of the maximum label value.
    probs.append(cum_probs[:, num_classes - 2])

    # Combine as columns into a new tensor.
    probs_tensor = tf.concat(tf.transpose(probs), axis=1)

    return probs_tensor
