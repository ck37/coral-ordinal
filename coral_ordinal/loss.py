from typing import Optional

import functools
import tensorflow as tf
import numpy as np


def _label_to_levels(labels: tf.Tensor, num_classes: int) -> tf.Tensor:
    # Original code that we are trying to replicate:
    # levels = [1] * label + [0] * (self.num_classes - 1 - label)
    # This function uses tf.sequence_mask(), which is vectorized. Avoids map_fn()
    # call.
    return tf.sequence_mask(labels, maxlen=num_classes - 1, dtype=tf.float32)


def _ordinal_loss_no_reduction(
    logits: tf.Tensor, levels: tf.Tensor, importance: tf.Tensor
) -> tf.Tensor:
    """Compute ordinal loss without reduction."""
    losses = -tf.reduce_sum(
        (
            tf.math.log_sigmoid(logits) * levels
            + (tf.math.log_sigmoid(logits) - logits) * (1.0 - levels)
        )
        * importance,
        axis=1,
    )
    return losses


# The outer function is a constructor to create a loss function using a certain number of classes.
@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class OrdinalCrossEntropy(tf.keras.losses.Loss):
    def __init__(
        self,
        num_classes: Optional[int] = None,
        importance_weights=None,
        from_type: str = "ordinal_logits",
        name: str = "ordinal_crossentropy",
        **kwargs,
    ):
        """Cross-entropy loss designed for ordinal outcomes.

        Args:
          num_classes: number of ranks (aka labels or values) in the ordinal variable.
            This is optional; can be inferred from size of y_pred at runtime.
          importance_weights: (Optional) importance weights for each binary classification task.
          from_type: one of "ordinal_logits" (default), "logits", or "probs".
            Ordinal logits are the output of a CoralOrdinal() layer with no activation.
            (Not yet implemented) Logits are the output of a dense layer with no activation.
            (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.
          name: name of layer
          **kwargs: keyword arguments passed to Loss().
        """
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.importance_weights = importance_weights
        self.from_type = from_type

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor):

        # Ensure that y_true is the same type as y_pred (presumably a float).
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        # Convert each true label to a vector of ordinal level indicators.
        tf_levels = _label_to_levels(tf.squeeze(y_true), self.num_classes)

        if self.importance_weights is None:
            importance_weights = tf.ones(self.num_classes - 1, dtype=tf.float32)
        else:
            importance_weights = tf.cast(self.importance_weights, dtype=tf.float32)

        if self.from_type == "ordinal_logits":
            loss = _ordinal_loss_no_reduction(y_pred, tf_levels, importance_weights)
        elif self.from_type == "probs":
            raise NotImplementedError("not yet implemented")
        elif self.from_type == "logits":
            raise NotImplementedError("not yet implemented")
        else:
            raise Exception(
                "Unknown from_type value "
                + self.from_type
                + " in OrdinalCrossEntropy()"
            )
        if self.reduction == tf.keras.losses.Reduction.NONE:
            return loss
        elif self.reduction in [
            tf.keras.losses.Reduction.AUTO,
            tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
        ]:
            return tf.reduce_mean(loss)
        elif self.reduction == tf.keras.losses.Reduction.SUM:
            return tf.reduce_sum(loss)
        else:
            raise Exception(f"{self.reduction} is not a valid reduction.")

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "importance_weights": self.importance_weights,
            "from_type": self.from_type,
        }
        base_config = super().get_config()
        return {**base_config, **config}
