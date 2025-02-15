from tensorflow.keras import layers, models
import tensorflow_hub as hub

class DeepMLP(models.Model):
    def __init__(self, input_size, num_classes):
        super(DeepMLP, self).__init__()

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


class BERTClassifier(models.Model):
    def __init__(self, bert_model_url, num_classes, hidden_dim=128, dropout_rate=0.3):
        super(BERTClassifier, self).__init__()

        self.bert_layer = hub.KerasLayer(bert_model_url, trainable=True)
        self.dense1 = layers.Dense(hidden_dim, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    def call(self, inputs):
        x = self.bert_layer(inputs)['pooled_output']
        x = self.dense1(x)
        x = self.dropout(x)
        return self.output_layer(x)