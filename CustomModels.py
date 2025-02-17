from tensorflow.keras import layers, models
import tensorflow_hub as hub

class DeepMLP_3(models.Model):
    def __init__(self, input_size, num_classes):
        super(DeepMLP_3, self).__init__()

        self.fc1 = layers.Dense(512, activation='relu', input_shape=(input_size,))
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(0.2)

        self.fc2 = layers.Dense(256, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(0.2)

        self.fc3 = layers.Dense(128, activation='relu')
        self.bn3 = layers.BatchNormalization()
        self.drop3 = layers.Dropout(0.2)

        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)

        x = self.fc3(x)
        x = self.bn3(x, training=training)
        x = self.drop3(x, training=training)

        return self.output_layer(x)



class DeepMLP_5(models.Model):
    def __init__(self, input_size, num_classes):
        super(DeepMLP_5, self).__init__()

        self.fc1 = layers.Dense(1024, activation='sigmoid', input_shape=(input_size,))
        self.bn1 = layers.BatchNormalization()
        self.drop1 = layers.Dropout(0.2)

        self.fc2 = layers.Dense(512, activation='sigmoid')
        self.bn2 = layers.BatchNormalization()
        self.drop2 = layers.Dropout(0.2)

        self.fc3 = layers.Dense(256, activation='sigmoid')
        self.bn3 = layers.BatchNormalization()
        self.drop3 = layers.Dropout(0.2)

        self.fc4 = layers.Dense(128, activation='sigmoid')
        self.bn4 = layers.BatchNormalization()
        self.drop4 = layers.Dropout(0.2)

        self.fc5 = layers.Dense(64, activation='sigmoid')
        self.bn5 = layers.BatchNormalization()
        self.drop5 = layers.Dropout(0.2)

        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs, training=False):
        x = self.fc1(inputs)
        x = self.bn1(x, training=training)
        x = self.drop1(x, training=training)

        x = self.fc2(x)
        x = self.bn2(x, training=training)
        x = self.drop2(x, training=training)

        x = self.fc3(x)
        x = self.bn3(x, training=training)
        x = self.drop3(x, training=training)

        x = self.fc4(x)
        x = self.bn4(x, training=training)
        x = self.drop4(x, training=training)

        x = self.fc5(x)
        x = self.bn5(x, training=training)
        x = self.drop5(x, training=training)

        return self.output_layer(x)



class GNN(models.Model):
    def __init__(self, input_size, hidden_dim, num_classes):
        super(GNN, self).__init__()

        self.fc1 = layers.Dense(hidden_dim, activation='relu', input_shape=(input_size,))
        self.fc2 = layers.Dense(hidden_dim, activation='relu')
        self.output_layer = layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        return self.output_layer(x)

