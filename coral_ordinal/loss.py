import tensorflow as tf
import tensorflow.python.ops as ops
import numpy as np

# The outer function is a constructor to create a loss function using a certain number of classes.
class OrdinalCrossEntropy(tf.keras.losses.Loss):
  
  def __init__(self,
               num_classes = None,
               importance = None,
               from_type = "ordinal_logits",
               name = "ordinal_crossent", **kwargs):
    """ Cross-entropy loss designed for ordinal outcomes.
    
    Args:
      num_classes: (Optional) how many ranks (aka labels or values) are in the ordinal variable.
        If not provided it will be automatically determined by on the prediction data structure.
      importance: (Optional) importance weights for each binary classification task.
      from_type: one of "ordinal_logits" (default), "logits", or "probs".
        Ordinal logits are the output of a CoralOrdinal() layer with no activation.
        (Not yet implemented) Logits are the output of a dense layer with no activation.
        (Not yet implemented) Probs are the probability outputs of a softmax or ordinal_softmax layer.
    """
    super(OrdinalCrossEntropy, self).__init__(name = name, **kwargs)
    
    self.num_classes = num_classes
      
    #self.importance_weights = importance
    if importance is None:
      self.importance_weights = tf.ones(self.num_classes - 1, dtype = tf.float32)
    else:
      self.importance_weights = tf.cast(importance, dtype = tf.float32)
      
    self.from_type = from_type


  # TODO: pre-compute so that this can be much faster on GPU.
  def label_to_levels(self, label):
    # Original code that we are trying to replicate:
    # levels = [1] * label + [0] * (self.num_classes - 1 - label)
    label_vec = tf.repeat(1, tf.cast(tf.squeeze(label), tf.int32))
    
    # This line requires that label values begin at 0. If they start at a higher
    # value it will yield an error.
    num_zeros = self.num_classes - 1 - tf.cast(tf.squeeze(label), tf.int32)
    
    zero_vec = tf.zeros(shape = (num_zeros), dtype = tf.int32)
    
    levels = tf.concat([label_vec, zero_vec], axis = 0)

    return tf.cast(levels, tf.float32)
    

  # Following https://www.tensorflow.org/api_docs/python/tf/keras/losses/Loss
  def call(self, y_true, y_pred):

    # Ensure that y_true is the same type as y_pred (presumably a float).
    y_pred = ops.convert_to_tensor_v2(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    
    if self.num_classes is None:
      # Determine number of classes based on prediction shape.
      if self.from_type == "ordinal_logits":
        # Number of classes = number of columns + 1
        self.num_classes = y_pred.shape[1] + 1
      else:
        self.num_classes = y_pred.shape[1]

    # Convert each true label to a vector of ordinal level indicators.
    # TODO: do this outside of the model, so that it's faster?
    tf_levels = tf.map_fn(self.label_to_levels, y_true)
    
    #if self.importance_weights is None:
      #self.importance_weights = np.ones(self.num_classes - 1, dtype = np.float32)
      #self.importance_weights = tf.ones(self.num_classes - 1, dtype = np.float32)
    
    if self.from_type == "ordinal_logits":
      #return ordinal_loss(y_pred, tf_levels, self.importance_weights)
      return ordinal_loss(y_pred, tf_levels, self.importance_weights)
    elif self.from_type == "probs":
      raise Exception("not yet implemented")
    elif self.from_type == "logits":
      raise Exception("not yet implemented")
    else:
      raise Exception("Unknown from_type value " + self.from_type +
                      " in OrdinalCrossEntropy()")
    
#def ordinal_loss(logits, levels, importance):
def ordinal_loss(logits, levels, importance):
    #levels = tf.cast(levels, tf.float32)
    val = (-tf.reduce_sum((tf.math.log_sigmoid(logits) * levels
                      #+ (tf.math.log_sigmoid(logits) - logits) * (1 - levels)) * tf.convert_to_tensor(importance, dtype = tf.float32),
                      + (tf.math.log_sigmoid(logits) - logits) * (1 - levels)) * importance,
           axis = 1))
    return tf.reduce_mean(val)
