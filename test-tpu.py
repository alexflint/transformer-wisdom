import argparse
import tensorflow as tf

@tf.function
def add_fn(x, y):
    z = x + y
    return z


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tpu", default="grpc://127.0.0.1:19870")
    args = parser.parse_args()
    print(args.tpu)

    print(f"connecting to TPU at {args.tpu}...")
    resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(tpu=args.tpu)

    print("creating strategy...")
    strategy = tf.distribute.TPUStrategy(resolver)

    print("running the operation...")
    x = tf.constant(1.)
    y = tf.constant(1.)
    z = strategy.run(add_fn, args=(x, y))

    print("done, printing result...")
    print(z)
    print("success")


if __name__ == "__main__":
    main()
