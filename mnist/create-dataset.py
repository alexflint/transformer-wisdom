import argparse
import tensorflow as tf

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", default="mnist_train")
    parser.add_argument("--test_data", default="mnist_test")
    args = parser.parse_args()

    tf.random.set_seed(SEED)

    print("loading dataset...")
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Add a channels dimension
    x_train = x_train[..., tf.newaxis].astype("float32")
    x_test = x_test[..., tf.newaxis].astype("float32")

    print("creating tf dataset...")
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    print("saving the training dataset...")
    tf.data.experimental.save(train_ds, args.train_data)

    print("saving the testing dataset...")
    tf.data.experimental.save(test_ds, args.test_data)


if __name__ == "__main__":
    main()
