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

from joblib import dump, load

import app.config as cf


from app.model.BERTSquad import BERTSquad
from app.model.squad_loss_fn import squad_loss_fn




if __name__ == "__main__":


    eval_examples = read_squad_examples(
        cf.INPUTS_FILE_DEV,
        is_training=False,
        version_2_with_negative=False
    )


    my_bert_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
        trainable=False
    )
    
    vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
    
    do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
    
    tokenizer = FullTokenizer(vocab_file, do_lower_case)


    dataset_size = convert_examples_to_features(
        examples=eval_examples,
        tokenizer=tokenizer,
        max_seq_length=384,
        doc_stride=128,
        max_query_length=64,
        is_training=False,
        output_fn=_append_feature,
        batch_size=4
    )

    my_bert_layer = hub.KerasLayer(
        "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
        trainable=False
    )
    
    vocab_file = my_bert_layer.resolved_object.vocab_file.asset_path.numpy()
    
    do_lower_case = my_bert_layer.resolved_object.do_lower_case.numpy()
    
    tokenizer = FullTokenizer(vocab_file, do_lower_case)

    eval_dataset = create_squad_dataset(
        cf.OUTPUT_EVAL_FILE,
        384,
        #input_meta_data['max_seq_length'],
        cf.BATCH_SIZE,
        is_training=False
    )


    RawResult = collections.namedtuple("RawResult", ["unique_id", "start_logits", "end_logits"])







