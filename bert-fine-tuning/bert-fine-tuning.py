# Tutorial: https://www.tensorflow.org/text/tutorials/fine_tune_bert

import os
import json
import numpy as np

import matplotlib

matplotlib.use('Agg')

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds

tfds.disable_progress_bar()

from official.modeling import tf_utils
from official import nlp
from official.nlp import bert

# Load the required submodules
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks


def encode_sentence(s, tokenizer):
    """Encode a sentence to a sequence of integers."""
    tokens = list(tokenizer.tokenize(s))
    tokens.append('[SEP]')
    return tokenizer.convert_tokens_to_ids(tokens)


def bert_encode(glue_dict, tokenizer):
    """Encode a split from the GLUE dataset to a tensorflow dataset."""
    sentence1 = tf.ragged.constant([
        encode_sentence(s, tokenizer) for s in np.array(glue_dict["sentence1"])
    ])
    sentence2 = tf.ragged.constant([
        encode_sentence(s, tokenizer) for s in np.array(glue_dict["sentence2"])
    ])

    cls = [tokenizer.convert_tokens_to_ids(['[CLS]'])] * sentence1.shape[0]
    input_word_ids = tf.concat([cls, sentence1, sentence2], axis=-1)

    input_mask = tf.ones_like(input_word_ids).to_tensor()

    type_cls = tf.zeros_like(cls)
    type_s1 = tf.zeros_like(sentence1)
    type_s2 = tf.ones_like(sentence2)
    input_type_ids = tf.concat([type_cls, type_s1, type_s2],
                               axis=-1).to_tensor()

    inputs = {
        'input_word_ids': input_word_ids.to_tensor(),
        'input_mask': input_mask,
        'input_type_ids': input_type_ids
    }

    return inputs


# unused for now
def load_and_evaluate_model(model_path, tokenizer, bert_classifier):
    my_examples = bert_encode(glue_dict={
        'sentence1': [
            'The rain in Spain falls mainly on the plain.',
            'Look I fine tuned BERT.'
        ],
        'sentence2': [
            'It mostly rains on the flat lands of Spain.',
            'Is it working? This does not match.'
        ]
    },
                              tokenizer=tokenizer)

    reloaded = tf.saved_model.load(model_path)
    reloaded_result = reloaded([
        my_examples['input_word_ids'], my_examples['input_mask'],
        my_examples['input_type_ids']
    ],
                               training=False)

    original_result = bert_classifier(my_examples, training=False)

    # The results are (nearly) identical:
    print(original_result.numpy())
    print()
    print(reloaded_result.numpy())


def main():
    # path to vocab, config, and parameters (must read from cloud storage for TPU)
    pretrained_bert_path = "gs://cloud-tpu-checkpoints/bert/v3/uncased_L-12_H-768_A-12"
    #pretrained_bert_path = "pretrained/uncased_L-12_H-768_A-12"

    vocab_file = os.path.join(pretrained_bert_path, "vocab.txt")
    bert_config_file = os.path.join(pretrained_bert_path, "bert_config.json")
    checkpoint_file = os.path.join(pretrained_bert_path, "bert_model.ckpt")

    # output for training, input for inference
    finetuned_model_path = "finetined/foo"

    #
    # Tokenizer
    #

    print("loading vocab...")
    tokenizer = bert.tokenization.FullTokenizer(vocab_file=vocab_file,
                                                do_lower_case=True)

    #
    # TPU
    #

    # connect and initialize the TPU
    print("connecting to TPU...")
    tpu_resolver = tf.distribute.cluster_resolver.TPUClusterResolver.connect(
        tpu='grpc://127.0.0.1:19870',
        zone='us-central1-b',
        project='transformer-wisdom')

    print("creating TPU strategy...")
    tpu_strategy = tf.distribute.TPUStrategy(tpu_resolver)

    print("running an operation on the TPU...")
    @tf.function
    def add_fn(x, y):
        z = x + y
        return z

    x = tf.constant(1.)
    y = tf.constant(1.)
    print(tpu_strategy.run(add_fn, args=(x, y)))

    with tpu_strategy.scope():

        #
        # Model
        #

        # load the config for the pre-trained BERT
        print("loading the config...")
        config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
        bert_config = bert.configs.BertConfig.from_dict(config_dict)
        print(bert_config)

        # create a keras BERT model using the config (weights will be randomly initialized)
        print("creating model...")
        bert_classifier, bert_encoder = bert.bert_models.classifier_model(
            bert_config, num_labels=2)

        # load the weights from the checkpoint
        checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
        checkpoint.read(checkpoint_file).assert_consumed()

        # plot the model architecture
        print("plotting model...")
        tf.keras.utils.plot_model(bert_classifier,
                                show_shapes=True,
                                dpi=48,
                                to_file="out/classifier.pdf")

        tf.keras.utils.plot_model(bert_encoder,
                                show_shapes=True,
                                dpi=48,
                                to_file="out/encoder.pdf")

        #
        # Dataset
        #

        # get the data for the "Microsoft Research Paraphrase Corpus" GLUE task
        # It's small, load the whole dataset
        glue, info = tfds.load('glue/mrpc', with_info=True, batch_size=-1)

        print(list(glue.keys()))
        print(info.features)

        # extract the training data
        glue_train = glue['train']

        # print the first training example
        for key, value in glue_train.items():
            print(f"{key:9s}: {value[0].numpy()}")

        print("vocab size:", len(tokenizer.vocab))

        # tokenize an example sentence
        tokens = tokenizer.tokenize("Hello TensorFlow!")
        print(tokens)
        ids = tokenizer.convert_tokens_to_ids(tokens)
        print(ids)

        # setup the training, validation, and test sets
        print("encoding dataset...")
        glue_train = bert_encode(glue['train'], tokenizer)
        glue_train_labels = glue['train']['label']

        glue_validation = bert_encode(glue['validation'], tokenizer)
        glue_validation_labels = glue['validation']['label']

        glue_test = bert_encode(glue['test'], tokenizer)
        glue_test_labels = glue['test']['label']

        print('glue_train shape:')
        for key, value in glue_train.items():
            print(f'  {key:15s} shape: {value.shape}')

        print(f'glue_train_labels shape: {glue_train_labels.shape}')

        #
        # Optimization
        #

        # set parameters for training
        epochs = 3
        batch_size = 32
        train_data_size = len(glue_train_labels)
        steps_per_epoch = int(train_data_size / batch_size)
        num_train_steps = steps_per_epoch * epochs
        warmup_steps = int(epochs * train_data_size * 0.1 / batch_size)

        # create an optimizer (this is still part of tf-models.official)
        optimizer = nlp.optimization.create_optimizer(
            2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

        assert isinstance(optimizer, official.nlp.optimization.AdamWeightDecay)

        # set up metrics and a loss function
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy('accuracy',
                                                    dtype=tf.float32)
        ]
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

        # compile the classifier (what does this actually do?)
        print("compiling model...")
        bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        # run training
        print("fitting model...")
        bert_classifier.fit(glue_train,
                            glue_train_labels,
                            validation_data=(glue_validation,
                                            glue_validation_labels),
                            batch_size=batch_size,
                            epochs=epochs)

        #
        # Export
        #

        print("exporting...")
        tf.saved_model.save(bert_classifier, export_dir=finetuned_model_path)


if __name__ == "__main__":
    main()
