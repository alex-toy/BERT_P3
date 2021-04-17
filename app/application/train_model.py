import tensorflow_hub as hub
import tensorflow as tf

from official.nlp.bert.tokenization import FullTokenizer
from official.nlp.bert.input_pipeline import create_squad_dataset
from official.nlp.data.squad_lib import generate_tf_record_from_json_file

from official.nlp import optimization

from official.nlp.data.squad_lib import read_squad_examples
from official.nlp.data.squad_lib import FeatureWriter
from official.nlp.data.squad_lib import convert_examples_to_features
from official.nlp.data.squad_lib import write_predictions

import numpy as np
import math
import random
import time
import json
import collections
import os

import app.config as cf


from app.model.BERTSquad import BERTSquad
from app.model.squad_loss_fn import squad_loss_fn



if __name__ == "__main__":

    input_meta_data = generate_tf_record_from_json_file(
        cf.INPUTS_FILE_TRAIN,
        cf.INPUTS_FILE_VOCAB,
        cf.INPUTS_FILE_DEV
    )

    with tf.io.gfile.GFile(cf.TRAIN_META_DATA, "w") as writer:
        writer.write(json.dumps(input_meta_data, indent=4) + "\n")


    train_dataset = create_squad_dataset(
        cf.INPUTS_FILE_DEV,
        input_meta_data['max_seq_length'], # 384
        cf.BATCH_SIZE,
        is_training=True
    )

    train_dataset_light = train_dataset.take(cf.NB_BATCHES_TRAIN)

    bert_squad = BERTSquad()

    optimizer = optimization.create_optimizer(
        init_lr=cf.INIT_LR,
        num_train_steps=cf.NB_BATCHES_TRAIN,
        num_warmup_steps=cf.WARMUP_STEPS
    )

    train_loss = tf.keras.metrics.Mean(name="train_loss")

    bert_squad.compile(
        optimizer,
        squad_loss_fn
    )

    ckpt = tf.train.Checkpoint(bert_squad=bert_squad)

    ckpt_manager = tf.train.CheckpointManager(ckpt, cf.CHECKPOINT_PATH, max_to_keep=1)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Last checkpoint restored!!")

