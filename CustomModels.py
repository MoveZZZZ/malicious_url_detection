from tensorflow.keras import layers, models
import tensorflow as tf
from tensorflow.keras import backend as K

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


class RBFLayer(layers.Layer):
    def __init__(self, num_centers, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.num_centers = num_centers

    def build(self, input_shape):
        self.centers = self.add_weight(name='centers',
                                       shape=(self.num_centers, input_shape[-1]),
                                       initializer='random_normal',
                                       trainable=True)
        self.betas = self.add_weight(name='betas',
                                     shape=(self.num_centers,),
                                     initializer='ones',
                                     trainable=True)
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - self.centers
        l2 = tf.reduce_sum(tf.square(diff), axis=-1)
        return tf.exp(-self.betas * l2)

class RBFN(models.Model):
    def __init__(self, num_classes, num_centers):
        super(RBFN, self).__init__()
        self.model = tf.keras.Sequential([
            RBFLayer(num_centers),
            layers.Dense(num_classes, activation='softmax')
        ])

    def call(self, inputs, training=False):
        return self.model(inputs)

class Optimization:
    @staticmethod
    def focal_loss(gamma=2., alpha=.25):
        def loss(y_true, y_pred):
            y_true = tf.keras.backend.cast(y_true, dtype=tf.float32)
            y_pred = tf.keras.backend.clip(y_pred, 1e-7, 1.0)
            y_true = tf.cond(
                tf.shape(y_true)[-1] < tf.shape(y_pred)[-1],
                lambda: tf.one_hot(tf.cast(tf.argmax(y_true, axis=-1), tf.int32), tf.shape(y_pred)[-1],
                                   dtype=tf.float32),
                lambda: y_true
            )
            pt = y_true * y_pred + (1 - y_true) * (1 - y_pred)
            loss = -alpha * tf.pow(1. - pt, gamma) * tf.math.log(pt + tf.keras.backend.epsilon())
            return tf.reduce_mean(loss)
        return loss
