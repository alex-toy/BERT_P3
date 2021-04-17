import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
INPUTS_FILE_DEV = os.path.join(INPUTS_FILE_PATH, 'dev-v1.1.json')
INPUTS_FILE_TRAIN = os.path.join(INPUTS_FILE_PATH, 'train-v1.1.json')
INPUTS_FILE_VOCAB = os.path.join(INPUTS_FILE_PATH, 'vocab.txt')


