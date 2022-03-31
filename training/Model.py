"""
Module to create a neural network model

This model only wraps a neural network model, for usage in this project.
It is outsourced, so that the implementation could easily exchanged by another model if necessary.

Authors: Dominik Brockmann, Niklas Schemmer
"""
from numpy import array, iterable
import tensorflow as tf
from tensorflow.keras import layers

class DenseModel(tf.keras.Model):
    """
    A dense neural network.

    Then network has flat outputs, which can be defined by 'num_actions'.
    Also the number of hidden_units can be defined by an iterable.
    """
    def __init__(self, hidden_units: iterable, num_actions: int):
        """
        Initialize the Model.

        Parameter hidden_units: an iterable of integers, which configure the amount of layers and its amount of neurons.
        Parameter num_actions: the amount of linear output parameters.
        """
        super(DenseModel, self).__init__()
        self.input_layer = layers.InputLayer()
        self.flatten_layer = layers.Flatten()
        self.hidden_layers = []

        # Iterate through hidden units to create hidden_layers
        for hidden_unit in hidden_units:
            self.hidden_layers.append(layers.Dense(hidden_unit, activation = 'tanh'))

        # Linear output for regression classifiers
        self.output_layer = layers.Dense(num_actions, activation = 'linear')

    @tf.function
    def call(self, inputs: array, **kwargs) -> array:
        """
        Calling the model on input parameters.

        Parameter inputs: array containing input batches that are of same shape as 'num_actions'.
        """
        x = self.input_layer(inputs)
        x = self.flatten_layer(x)

        # Iterately call hidden_layers
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)

        return output

    def copy_weights_to(self, copy_to: tf.keras.Model):
        """
        This copies the weights to another network.

        Parameter copy_to: The model the weights are copied to
        """
        variables2 = self.trainable_variables
        variables1 = copy_to.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())