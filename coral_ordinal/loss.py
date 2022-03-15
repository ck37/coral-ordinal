from typing import Optional

import warnings
import functools
import tensorflow as tf
import numpy as np


def _label_to_levels(labels: tf.Tensor, num_classes: int) -> tf.Tensor:
    # Original code that we are trying to replicate:
    # levels = [1] * label + [0] * (self.num_classes - 1 - label)
    # This function uses tf.sequence_mask(), which is vectorized. Avoids map_fn()
    # call.
    return tf.sequence_mask(labels, maxlen=num_classes - 1, dtype=tf.float32)


def _coral_ordinal_loss_no_reduction(
    logits: tf.Tensor, levels: tf.Tensor, importance: tf.Tensor
) -> tf.Tensor:
    """Compute ordinal loss without reduction."""
    levels = tf.cast(levels, dtype=logits.dtype)
    importance = tf.cast(importance, dtype=logits.dtype)
    losses = -tf.reduce_sum(
        (
            tf.math.log_sigmoid(logits) * levels
            + (tf.math.log_sigmoid(logits) - logits) * (1.0 - levels)
        )
        * importance,
        axis=1,
    )
    return losses


def _reduce_losses(
    losses: tf.Tensor, reduction: tf.keras.losses.Reduction
) -> tf.Tensor:
    """Reduces losses to specified reduction."""
    if reduction == tf.keras.losses.Reduction.NONE:
        return losses
    elif reduction in [
        tf.keras.losses.Reduction.AUTO,
        tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
    ]:
        return tf.reduce_mean(losses)
    elif reduction == tf.keras.losses.Reduction.SUM:
        return tf.reduce_sum(losses)
    else:
        raise Exception(f"{reduction} is not a valid reduction.")


# The outer function is a constructor to create a loss function using a certain number of classes.
@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class OrdinalCrossEntropy(tf.keras.losses.Loss):
    """Computes ordinal cross entropy based on ordinal predictions and outcomes."""

    def __init__(
        self,
        num_classes: Optional[int] = None,
        importance_weights: Optional[np.ndarray] = None,
        from_type: str = "ordinal_logits",
        name: str = "ordinal_crossentropy",
        **kwargs,
    ):
        """Cross-entropy loss designed for ordinal outcomes.

        Args:
          num_classes: number of ranks (aka labels or values) in the ordinal variable.
            This is optional; can be inferred from size of y_pred at runtime.
          importance_weights: class weights for each binary classification task.
          from_type: one of "ordinal_logits" (default), "logits", or "probs".
            Ordinal logits are the output of a CoralOrdinal() layer with no activation.
            (Not yet implemented) Logits are the output of a dense layer with no activation.
            (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax
            layer.
          name: name of layer
          **kwargs: keyword arguments passed to Loss().
        """
        super().__init__(name=name, **kwargs)

        self.num_classes = num_classes
        self.importance_weights = importance_weights
        self.from_type = from_type

    # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
    def call(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ):

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
            losses = _coral_ordinal_loss_no_reduction(
                y_pred, tf_levels, importance_weights
            )
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

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, losses.dtype)
            losses = tf.multiply(losses, sample_weight)

        return _reduce_losses(losses, self.reduction)

    def get_config(self):
        config = {
            "num_classes": self.num_classes,
            "importance_weights": self.importance_weights,
            "from_type": self.from_type,
        }
        base_config = super().get_config()
        return {**base_config, **config}


@tf.keras.utils.register_keras_serializable(package="coral_ordinal")
class CornOrdinalCrossEntropy(tf.keras.losses.Loss):
    """Implements CORN ordinal loss function for logits."""

    def __init__(
        self,
        **kwargs,
    ):
        """Initializes class."""
        super().__init__(**kwargs)
        self.num_classes = None

    def __call__(
        self,
        y_true: tf.Tensor,
        y_pred: tf.Tensor,
        sample_weight: Optional[tf.Tensor] = None,
    ):
        """Computes the CORN loss described in https://arxiv.org/abs/2111.08851

        Args:
          y_true: true labels (0..N-1)
          y_pred: predicted logits (from CornLayer())
          sample_weights: optional; provide sample weights for each sample.

        Returns:
          loss: tf.Tensor, that contains the loss value.
        """
        y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = tf.squeeze(y_true)
        if self.num_classes is None:
            self.num_classes = int(y_pred.get_shape().as_list()[1]) + 1

        sets = []
        for i in range(self.num_classes - 1):
            set_mask = y_true > i - 1
            label_gt_i = y_true > i
            sets.append((set_mask, label_gt_i))

        n_examples = []
        losses = tf.zeros_like(y_true)
        for task_index, s in enumerate(sets):
            set_mask, label_gt_i = s
            n_examples_task = tf.reduce_sum(tf.cast(set_mask, dtype=tf.float32))
            n_examples.append(n_examples_task)

            pred_task = tf.gather(y_pred, task_index, axis=1)
            losses_task = tf.where(
                set_mask,
                tf.where(
                    label_gt_i,
                    tf.math.log_sigmoid(pred_task),  # if label > i
                    tf.math.log_sigmoid(pred_task) - pred_task,  # if label <= i
                ),
                0.0,  # don't add to loss if label is <= i - 1
            )
            losses += -losses_task
        losses /= self.num_classes

        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, losses.dtype)
            losses = tf.multiply(losses, sample_weight)

        return _reduce_losses(losses, self.reduction)
