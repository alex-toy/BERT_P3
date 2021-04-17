import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

INPUTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data'))
INPUTS_FILE_DEV = os.path.join(INPUTS_FILE_PATH, 'dev-v1.1.json')
INPUTS_FILE_TRAIN = os.path.join(INPUTS_FILE_PATH, 'train-v1.1.json')
INPUTS_FILE_VOCAB = os.path.join(INPUTS_FILE_PATH, 'vocab.txt')
TRAIN_META_DATA = os.path.join(INPUTS_FILE_PATH, 'train_meta_data')

CHECKPOINT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../checkpoint'))
OUTPUTS_MODELS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../models'))


OUTPUTS_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../output'))
OUTPUT_PRED_FILE = os.path.join(OUTPUTS_FILE_PATH, 'predictions.json')
OUTPUT_NBEST_FILE = os.path.join(OUTPUTS_FILE_PATH, 'nbest_predictions.json')
OUTPUT_NULL_LOG_ODDS_FILE = os.path.join(OUTPUTS_FILE_PATH, 'null_odds.json')


BATCH_SIZE = 4
TRAIN_DATA_SIZE = 88641
NB_BATCHES_TRAIN = 2000
NB_EPOCHS = 3
INIT_LR = 5e-5
WARMUP_STEPS = int(NB_BATCHES_TRAIN * 0.1)



