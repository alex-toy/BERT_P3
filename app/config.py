import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
INPUTS_FILE_DEV = os.path.join(INPUTS_FILE_PATH, 'dev-v1.1.json')
INPUTS_FILE_TRAIN = os.path.join(INPUTS_FILE_PATH, 'train-v1.1.json')
INPUTS_FILE_VOCAB = os.path.join(INPUTS_FILE_PATH, 'vocab.txt')
TRAIN_META_DATA = os.path.join(INPUTS_FILE_PATH, 'train_meta_data')

CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../checkpoint'))

BATCH_SIZE = 4
TRAIN_DATA_SIZE = 88641
NB_BATCHES_TRAIN = 2000
BATCH_SIZE = 4
NB_EPOCHS = 3
INIT_LR = 5e-5
WARMUP_STEPS = int(NB_BATCHES_TRAIN * 0.1)


