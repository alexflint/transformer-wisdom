import tensorflow as tf


@tf.function
def add_fn(x, y):
    z = x + y
    return z


def main():
    print("creating the resolver...")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
        tpu='grpc://127.0.0.1:19870',
        zone='us-central1-b',
        project='transformer-wisdom')

    print("creating strategy...")
    strategy = tf.distribute.TPUStrategy(resolver)

    print("running the operation...")
    x = tf.constant(1.)
    y = tf.constant(1.)
    z = strategy.run(add_fn, args=(x, y))

    print("done, printing result...")
    print(z)


if __name__ == "__main__":
    main()
