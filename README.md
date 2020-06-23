# Ordinal regression in Tensorflow Keras


Tensorflow Keras implementation of ordinal regression (aka ordinal classification) using consistent rank logits (CORAL) by Cao, Mirjalili, & Raschka (2019).

This package includes:

  * Ordinal output layer: `CoralOrdinal()`
  * Ordinal loss function: `OrdinalCrossEntropy()`
  * Ordinal error metric: `MeanAbsoluteErrorLabels()`
  * Ordinal activation function: `logits_to_probs()`

This is a work in progress, so please post any issues to the [issue queue](https://github.com/ck37/coral-ordinal/issues).

**Acknowledgments**: Many thanks to [Sebastian Raschka](https://github.com/rasbt) for the help in porting from the [PyTorch source repository](https://github.com/Raschka-research-group/coral-cnn/).

Key pending items:

  * Function docstrings
  * Docs
  * Tests
  * Custom metrics: accuracy, cross-entropy

## Installation

Install the stable version via pip:

```bash
pip install coral-ordinal
```

Install the most recent code on GitHub via pip:

```bash
pip install git+https://github.com/ck37/coral-ordinal/
```

## Dependencies

This package relies on Python 3.6+, Tensorflow 2.2+, numpy, pandas, and scipy.

## Example

[See this colab notebook](https://colab.research.google.com/drive/1AQl4XeqRRhd7l30bmgLVObKt5RFPHttn) for examples of ordinal regression with MNIST (multilayer perceptron) and Amazon reviews (universal sentence encoder).

## References

Cao, W., Mirjalili, V., & Raschka, S. (2019). [Consistent rank logits for ordinal regression with convolutional neural networks]( https://arxiv.org/abs/1901.07884). arXiv preprint arXiv:1901.07884, 6. 
