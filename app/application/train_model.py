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



if __name__ == "__main__":

    input_meta_data = generate_tf_record_from_json_file(
        cf.INPUTS_FILE_DEV,
        cf.INPUTS_FILE_VOCAB,
        cf.INPUTS_FILE_TRAIN
    )

