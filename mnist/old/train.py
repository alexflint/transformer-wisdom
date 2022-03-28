import argparse
import numpy as np

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# Random seed for determinism when shuffling the dataset
SEED = 123

# Global batch size, will be split among 8 TPU cores
BATCH_SIZE = 64

# Number of epochs of training
EPOCHS = 2

class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(32, 3, activation='relu')
    self.flatten = Flatten()
    self.d1 = Dense(128, activation='relu')
    self.d2 = Dense(10)

  def call(self, x):
    x = self.conv1(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", default="grpc://127.0.0.1:19870")
    parser.add_argument("--train_data", default="gs://transformer-wisdom-data/mnist-train")
    parser.add_argument("--test_data", default="gs://transformer-wisdom-data/mnist-test")
    args = parser.parse_args()

    print(f"connecting to TPU at {args.tpu}...")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=args.tpu)
    strategy = tf.distribute.TPUStrategy(resolver)

    # initialize the random seed for determinism
    #tf.random.set_seed(SEED)

    print("loading the dataset...")
    train_ds = tf.data.experimental.load(args.train_data).shuffle(10000, seed=SEED).batch(BATCH_SIZE)
    test_ds = tf.data.experimental.load(args.test_data).batch(BATCH_SIZE)

    # Create an instance of the model
    print("creating model, loss, optimizer...")
    model = MyModel()
    optimizer = tf.keras.optimizers.Adam()

    with strategy.scope():
        loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True,
            reduction=tf.keras.losses.Reduction.NONE)   # NONE means we get per-example losses

        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        def compute_loss(labels, predictions):
            losses = loss_obj(labels, predictions)
            return tf.nn.compute_average_loss(losses, global_batch_size=BATCH_SIZE)


    def train_step(images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

        return loss


    def test_step(images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = model(images, training=False)
        t_loss = compute_loss(labels, predictions)

        test_loss(t_loss)
        test_accuracy(labels, predictions)


    @tf.function
    def distributed_train_step(images, labels):
        per_replica_losses = strategy.run(train_step, args=(images, labels))
        return strategy.reduce(
            tf.distribute.ReduceOp.SUM,
            per_replica_losses,
            axis=None)

    @tf.function
    def distributed_test_step(images, labels):
        return strategy.run(test_step, args=(images, labels))

    print(f"running {EPOCHS} epochs of training...")
    for epoch in range(EPOCHS):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_accuracy.reset_states()
        test_loss.reset_states()
        test_accuracy.reset_states()

        for images, labels in train_ds:
            distributed_train_step(images, labels)

        for test_images, test_labels in test_ds:
            distributed_test_step(test_images, test_labels)

        print(
            f'Epoch {epoch + 1}, '
            f'Loss: {train_loss.result()}, '
            f'Accuracy: {train_accuracy.result() * 100}, '
            f'Test Loss: {test_loss.result()}, '
            f'Test Accuracy: {test_accuracy.result() * 100}'
        )

if __name__ == "__main__":
    main()
    print("done")