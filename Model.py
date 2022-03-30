import tensorflow as tf
from tensorflow.keras import layers

class DenseModel(tf.keras.Model):
    def __init__(self, hidden_units, num_actions):
        super(DenseModel, self).__init__()
        self.input_layer = layers.InputLayer()
        self.hidden_layers = []

        for hidden_unit in hidden_units:
            self.hidden_layers.append(layers.Dense(hidden_unit, activation = 'tanh'))

        self.output_layer = layers.Dense(num_actions, activation = 'linear')

    @tf.function
    def call(self, inputs, **kwargs):
        x = self.input_layer(inputs)
        for layer in self.hidden_layers:
            x = layer(x)
        output = self.output_layer(x)
        return output
