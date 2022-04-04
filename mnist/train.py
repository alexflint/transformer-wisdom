import os
import sys
import argparse

import glog as log
import numpy as np
import tensorflow as tf

# Random seed for determinism when shuffling the dataset
SEED = 123

# Pre-replica batch size, will be split among 8 TPU cores
BATCH_SIZE_PER_REPLICA = 64


def print(s):
    __builtins__.print(s)
    sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", default="grpc://127.0.0.1:19870")
    parser.add_argument("--train_data",
                        default="gs://transformer-wisdom-data/mnist-train")
    parser.add_argument("--test_data",
                        default="gs://transformer-wisdom-data/mnist-test")
    parser.add_argument("--checkpoints",
                        default="gs://transformer-wisdom-data/experiment0")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    log.info(f"connecting to TPU at {args.tpu}...")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
        tpu=args.tpu)
    strategy = tf.distribute.TPUStrategy(resolver)
    log.info(f"number of devices: {strategy.num_replicas_in_sync}")

    # initialize the random seed for determinism
    #tf.random.set_seed(SEED)

    global_batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    #
    # Load the data
    #

    log.info("loading the dataset...")
    with strategy.scope():
        train_data = tf.data.experimental.load(args.train_data).shuffle(
            10000, seed=SEED).batch(global_batch_size)
        test_data = tf.data.experimental.load(
            args.test_data).batch(global_batch_size)

    distr_train_data = strategy.experimental_distribute_dataset(train_data)
    distr_test_data = strategy.experimental_distribute_dataset(test_data)

    # Create a checkpoint directory
    checkpoint_prefix = os.path.join(args.checkpoints, "ckpt")

    #
    # Create model
    #

    log.info("creating model....")
    with strategy.scope():
        # Set reduction to `none` so we can do the reduction afterwards and divide by
        # global batch size.
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

        def compute_loss(labels, predictions):
            per_example_loss = loss_object(labels, predictions)
            return tf.nn.compute_average_loss(
                per_example_loss, global_batch_size=global_batch_size)

        test_loss = tf.keras.metrics.Mean(name='test_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='train_accuracy')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(
            name='test_accuracy')

        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        optimizer = tf.keras.optimizers.Adam()

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

    # Training step
    def train_step(inputs):
        images, labels = inputs

        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = compute_loss(labels, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        train_accuracy.update_state(labels, predictions)
        return loss

    # Test step
    def test_step(inputs):
        images, labels = inputs

        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)

        test_loss.update_state(t_loss)
        test_accuracy.update_state(labels, predictions)

    # `run` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.run(train_step, args=(dataset_inputs, ))
        return strategy.reduce(tf.distribute.ReduceOp.SUM,
                               per_replica_losses,
                               axis=None)

    @tf.function
    def distributed_test_step(dataset_inputs):
        return strategy.run(test_step, args=(dataset_inputs, ))

    # print model summary
    model.summary()

    print(f"training model for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in distr_train_data:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches

        # TEST LOOP
        for x in distr_test_data:
            distributed_test_step(x)

        if epoch % 2 == 0:
            log.info(f"saving checkpoint to {checkpoint_prefix}...")
            checkpoint.save(checkpoint_prefix)

        template = "Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}"
        log.info(
            template.format(epoch + 1, train_loss,
                            train_accuracy.result() * 100, test_loss.result(),
                            test_accuracy.result() * 100))

        test_loss.reset_states()
        train_accuracy.reset_states()
        test_accuracy.reset_states()


if __name__ == "__main__":
    main()
