import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers

from tensorflow.python.keras import activations

from .loss import OrdinalCrossEntropy

class CoralOrdinal(tf.keras.layers.Layer):

  # We skip input_dim/input_shape here and put in the build() method as recommended in the tutorial,
  # in case the user doesn't know the input dimensions when defining the model.
  def __init__(self, num_classes, activation = None, **kwargs):
    """ Ordinal output layer, which produces ordinal logits by default.
    
    Args:
      num_classes: how many ranks (aka labels or values) are in the ordinal variable.
      activation: (Optional) Activation function to use. The default of None produces
        ordinal logits, but passing "ordinal_softmax" will cause the layer to output
        a probability prediction for each label.
    """
    
    # Via Dense Layer code:
    # https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/layers/core.py#L1128
    if 'input_shape' not in kwargs and 'input_dim' in kwargs:
      kwargs['input_shape'] = (kwargs.pop('input_dim'),)

    # Pass any additional keyword arguments to Layer() (i.e. name, dtype)
    super(CoralOrdinal, self).__init__(**kwargs)
    self.num_classes = num_classes
    self.activation = activations.get(activation)
    
  # Following https://www.tensorflow.org/guide/keras/custom_layers_and_models#best_practice_deferring_weight_creation_until_the_shape_of_the_inputs_is_known
  def build(self, input_shape):

    # Single fully-connected neuron - this is the latent variable.
    num_units = 1

    # I believe glorot_uniform (aka Xavier uniform) is pytorch's default initializer, per
    # https://pytorch.org/docs/master/generated/torch.nn.Linear.html
    # and https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotUniform
    self.fc = self.add_weight(shape = (input_shape[-1], num_units),
                              initializer = 'glorot_uniform',
                              # Not sure if this is necessary:
                              dtype = tf.float32,
                              trainable = True)
                              
    # num_classes - 1 bias terms, defaulting to 0.
    self.linear_1_bias = self.add_weight(shape = (self.num_classes - 1, ),
                                         initializer = 'zeros',
                                         # Not sure if this is necessary:
                                         dtype = tf.float32,
                                         trainable = True)

  # This defines the forward pass.
  def call(self, inputs):
    fc_inputs = tf.matmul(inputs, self.fc)

    logits = fc_inputs + self.linear_1_bias
    
    if self.activation is None:
      outputs = logits
    else:
      # Not yet tested:
      outputs = self.activation(logits)

    return outputs
  
  # This allows for serialization supposedly.
  # https://www.tensorflow.org/guide/keras/custom_layers_and_models#you_can_optionally_enable_serialization_on_your_layers
  def get_config(self):
    config = super(CoralOrdinal, self).get_config()
    config.update({'num_classes': self.num_classes})
    return config
