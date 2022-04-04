import os
import sys
import argparse

import glog as log
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

# Random seed for determinism when shuffling the dataset
SEED = 123

# Pre-replica batch size, based on TPUv2 hardware spec
BATCH_SIZE_PER_REPLICA = 128


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", default="grpc://127.0.0.1:19870")
    parser.add_argument("--train_data",
                        default="gs://transformer-wisdom-data/mnist-train")
    parser.add_argument("--test_data",
                        default="gs://transformer-wisdom-data/mnist-test")
    parser.add_argument("--checkpoints",
                        default="gs://transformer-wisdom-data/experiment1")
    parser.add_argument(
        "--final_model",
        default="gs://transformer-wisdom-data/experiment1/final")
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    log.info(f"connecting to TPU at {args.tpu}...")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
        tpu=args.tpu)
    strategy = tf.distribute.TPUStrategy(resolver)
    log.info(f"number of devices: {strategy.num_replicas_in_sync}")

    # initialize the random seed for determinism
    #tf.random.set_seed(SEED)

    #
    # Load the data
    #

    batch_size = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

    def load_dataset(split):
        dataset, info = tfds.load(name='mnist',
                                  split=split,
                                  with_info=True,
                                  as_supervised=True,
                                  try_gcs=True)

        # Normalize the input data.
        def scale(image, label):
            image = tf.cast(image, tf.float32)
            image /= 255.0
            return image, label

        dataset = dataset.map(scale)

        # Only shuffle and repeat the dataset in training. The advantage of having an
        # infinite dataset for training is to avoid the potential last partial batch
        # in each epoch, so that you don't need to think about scaling the gradients
        # based on the actual batch size.
        if split == 'train':
            dataset = dataset.shuffle(10000, seed=SEED)
            dataset = dataset.repeat()

        return dataset.batch(batch_size), info.splits[split]

    train_data, train_data_info = load_dataset('train')
    test_data, test_data_info = load_dataset('test')

    #
    # Create model
    #

    log.info("creating the model....")
    with strategy.scope():
        log.info("going....")
        train_loss = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=True)

        log.info("going....")
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10)
        ])

        log.info("going....")
        #checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)

        log.info(f"training for {args.epochs} epochs...")
        model.compile(loss=train_loss,
                      optimizer='adam',
                      metrics=['sparse_categorical_accuracy'])

    #
    # Print the model summary
    #

    #
    # Train the model
    #
    model.fit(train_data,
              epochs=args.epochs,
              steps_per_epoch=train_data_info.num_examples // batch_size,  # required because the dataset uses repeat()
              validation_steps=test_data_info.num_examples // batch_size,  # required because the dataset uses repeat()
              validation_data=test_data)

    # print a table to output
    model.summary()

    # save the model weights and layout
    model.save(args.final_model)
    log.info(f"saved model to {args.final_model}")

    # TODO: save checkpoints, add metrics


if __name__ == "__main__":
    main()
