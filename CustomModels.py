import tensorflow as tf
from tensorflow.keras import layers, models

class DeepMLP_3(models.Model):
    def __init__(self, input_size, num_classes, _activation, **kwargs):
        super(DeepMLP_3, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self._activation = _activation
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
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(DeepMLP_3, self).build(input_shape)
        self.model.build(input_shape)
        print("DeepMLP_3 built with input shape:", input_shape)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DeepMLP_5(models.Model):
    def __init__(self, input_size, num_classes, _activation, **kwargs):
        super(DeepMLP_5, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self._activation = _activation
        self.model = tf.keras.Sequential([
            layers.Dense(1024, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(512, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax")
        ])
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(DeepMLP_5, self).build(input_shape)
        self.model.build(input_shape)
        print("DeepMLP_5 built with input shape:", input_shape)

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class DeepMLP_7(models.Model):
    def __init__(self, input_size, num_classes, _activation, **kwargs):
        super(DeepMLP_7, self).__init__(**kwargs)
        self.input_size = input_size
        self.num_classes = num_classes
        self._activation = _activation
        self.model = tf.keras.Sequential([
            layers.Dense(4096, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(2048, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(1024, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(512, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(256, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(128, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(64, activation=_activation),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax")
        ])
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(DeepMLP_7, self).build(input_shape)
        self.model.build(input_shape)
        print("DeepMLP_7 built with input shape:", input_shape)
    def call(self, inputs, training=False):
        return self.model(inputs, training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class GNN(models.Model):
    def __init__(self, input_size, hidden_dim, num_classes, _activation, **kwargs):
        super(GNN, self).__init__(**kwargs)
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self._activation = _activation
        self.model = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation=_activation, input_shape=(input_size,)),
            layers.Dense(hidden_dim, activation=_activation),
            layers.Dense(num_classes, activation="softmax")
        ])
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(GNN, self).build(input_shape)
        self.model.build(input_shape)
        print("GNN built with input shape:", input_shape)
    def call(self, inputs):
        return self.model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "hidden_dim": self.hidden_dim,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class RBFLayer(layers.Layer):
    def __init__(self, num_centers, centers_init=None, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.num_centers = num_centers
        self.centers_init = centers_init

    def build(self, input_shape):
        if self.centers_init is not None:
            self.centers = self.add_weight(
                name='centers',
                shape=(self.num_centers, input_shape[-1]),
                initializer=tf.constant_initializer(self.centers_init),
                trainable=True
            )
        else:
            self.centers = self.add_weight(
                name='centers',
                shape=(self.num_centers, input_shape[-1]),
                initializer='random_normal',
                trainable=True
            )
        self.betas = self.add_weight(
            name='betas',
            shape=(self.num_centers,),
            initializer='ones',
            trainable=True
        )
        super(RBFLayer, self).build(input_shape)

    def call(self, inputs):
        diff = tf.expand_dims(inputs, axis=1) - self.centers
        l2 = tf.reduce_sum(tf.square(diff), axis=-1)
        return tf.exp(-self.betas * l2)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_centers": self.num_centers,
            "centers_init": self.centers_init
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class RBFN(models.Model):
    def __init__(self, num_classes, num_centers, centers_init=None, **kwargs):
        super(RBFN, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.num_centers = num_centers
        self.centers_init = centers_init
        self.model = tf.keras.Sequential([
            RBFLayer(num_centers, centers_init=centers_init),
            layers.Dense(num_classes, activation='softmax')
        ])

    def build(self, input_shape):
        super(RBFN, self).build(input_shape)
        self.model.build(input_shape)
        print("RBFN built with input shape:", input_shape)

    def call(self, inputs, training=False):
        return self.model(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_classes": self.num_classes,
            "num_centers": self.num_centers,
            "centers_init": self.centers_init
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class AutoencoderClassifier(models.Model):
    def __init__(self, input_size, encoding_dim, num_classes, _activation='relu', **kwargs):
        super(AutoencoderClassifier, self).__init__(**kwargs)
        self.input_size = input_size
        self.encoding_dim = encoding_dim
        self.num_classes = num_classes
        self._activation = _activation
        self.encoder = tf.keras.Sequential([
            layers.Dense(256, activation=_activation, input_shape=(input_size,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(encoding_dim, activation=_activation)
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation=_activation, input_shape=(encoding_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(input_size, activation='sigmoid')
        ])
        self.classifier = tf.keras.Sequential([
            layers.Dense(128, activation=_activation, input_shape=(encoding_dim,)),
            layers.BatchNormalization(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation='softmax')
        ])
        self.build(input_shape=(None, input_size))

    def build(self, input_shape):
        super(AutoencoderClassifier, self).build(input_shape)
        encoded_shape = self.encoder.compute_output_shape(input_shape)
        print("AutoencoderClassifier built with input shape:", input_shape,
              "and encoder output shape:", encoded_shape)
    def call(self, inputs, training=False):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        classification = self.classifier(encoded)
        return {"decoder": decoded, "classifier": classification}

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_size": self.input_size,
            "encoding_dim": self.encoding_dim,
            "num_classes": self.num_classes,
            "_activation": self._activation
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

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

    @staticmethod
    def weighted_categorical_crossentropy(class_weights):
        def loss(y_true, y_pred):
            y_true = tf.keras.backend.cast(y_true, dtype=tf.float32)
            weights = tf.reduce_sum(class_weights * y_true, axis=-1, keepdims=True)
            cross_entropy = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
            return tf.keras.backend.mean(weights * cross_entropy)
        return loss