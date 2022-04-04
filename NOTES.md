# Notes

## Cloud build versus TPU VM

Things are still orders of magnitude slower when running on cloud build versus directly on the TPU VM. A build run took 10 hours to perform 175 epochs of MNIST, whereas on the TPU VM the same takes about 15 minutes. That's a 40x difference.

This guide to running Keras models on TPU from 2019 provides a very succinct example: https://blog.tensorflow.org/2019/01/keras-on-tpus-in-colab.html. But it uses keras_to_tpu_model which was apparently deprecated later in 2019.

This general guide to distrsibuted training contains a lot of info about TPUs: https://www.tensorflow.org/guide/distributed_training

It seems that the basic choices for the training loop are keras.model.fit, estimator API, or a custom training loop. The Estimator API seems to be deprecated.

The basic elements are variables, layers, models, optimizers, metrics, summaries, and checkpoints.

Batch size of 128 is best for v2 TPUs, and since each TPU has 8 cores, use global batch size of 1024

## Training strategies

The distribution strategy TPUStrategy works by running different training examples on different TPU cores, then applying identical gradient steps on all TPU cores. Hence all TPU cores contain an identical complete model, and they all run in sync with each other.

The main other distribution strategies are MirroredStrategy (very similar to TPUStrategy, but for multiple GPUs on a single machine) and MultiWorkerMirroredStrategy (very similar again, but for multiple GPUs across multiple machines)

There is also OneDeviceStrategy, which puts all computation on a single GPU or TPU

You can get the current strategy with tf.distribute.get_strategy 

The non-distributed strategy can be obtained via tf.distribute.get_strategy() outside of any strategy.scope

Strategies can run things like this:
    strategy.run(fn)

Strategies can reduce things like this:
    strategy.reduce("SUM", 1., axis=None)  # reduce some values

## Using keras.model.fit

```python
mirrored_strategy = tf.distribute.MirroredStrategy()

with mirrored_strategy.scope():
  model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])

model.compile(loss='mse', optimizer='sgd')

dataset = tf.data.Dataset.from_tensors(([1.], [1.])).repeat(100).batch(10)
model.fit(dataset, epochs=2)
```

## Saving model architectures

Use model.to_json and keras.models.model_from_json(json_string, custom_objects={})

## Saving model weights

There are two ways: keras.Model.save_weights and tf.train.Checkpoint.save. The docs
say to prefer checkpoints for saving training weights. They use similar formats but
are not compatible.
