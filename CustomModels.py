from tensorflow.keras import layers, models
import tensorflow as tf

class DeepMLP_3(models.Model):
    def __init__(self, input_size, num_classes, _activation):
        super(DeepMLP_3, self).__init__()

        self.model = tf.keras.Sequential([
            layers.Dense(512, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(256, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(128, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(num_classes, activation="softmax")
        ])
        self.model.build((None, input_size))

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)



class DeepMLP_5(models.Model):
    def __init__(self, input_size, num_classes, _activation):
        super(DeepMLP_5, self).__init__()

        self.model = tf.keras.Sequential([
            layers.Dense(1024, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(512, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(256, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(128, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(64, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),

            layers.Dense(num_classes, activation="softmax")
        ])

        self.model.build((None, input_size))
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)



class GNN(models.Model):
    def __init__(self, input_size, hidden_dim, num_classes, _activation):
        super(GNN, self).__init__()

        self.model = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation=_activation, input_shape=(input_size,)),
            layers.Dense(hidden_dim, activation=_activation),
            layers.Dense(num_classes, activation="softmax")
        ])
        self.model.build((None, input_size))

    def call(self, inputs):
        return self.model(inputs)

