from pathlib import Path

WRITE_LOGS_TO_GOOGLE_DRIVE = False
DEVICE = 'cpu'

# Parent to all other diretories
BASE_DIR = Path('/Volumes/MyPassport/Data/NameGen/')

# Input datasets
DATA_DIR = BASE_DIR / 'Data'

# Different files that define the state of a model
# (some of them are updated at each iteration) and
# allow us to restore everything in case of failure.
STATE_DIR = BASE_DIR / 'State'

# Stores vocabularies constructed from the training data.
# They are used to encode the textual input to numbers.
# Without them it is impossible to use the trained model
LANGS_DIR = STATE_DIR / 'Langs'

# Stores dataset split into training, validation, and test
# subsets. It is important to preserve at least the test
# dataset because after the training is complete we need
# to know which methods the model was not trained on
TRAIN_VALID_TEST_DIR = STATE_DIR / 'Train-Valid-Test'

# Training logs
LOGS_DIR = BASE_DIR / 'Logs'

# Generated names for test and validation sets and history of scores
RESULTS_DIR = BASE_DIR / 'Results'

# Visualizations created during training
IMG_DIR = BASE_DIR / 'Img'

# Trained model (updated at each iteration)
MODELS_DIR = BASE_DIR / 'Models'

SOS_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

MAX_LENGTH = 50

TRAIN_PROP = 0.7
VALID_PROP = 0.1
TEST_PROP = 0.2

HIDDEN_SIZE = 256
LEARNING_RATE = 0.01
TEACHER_FORCING_RATIO = 0.5

NUM_ITER = 100000
LOG_EVERY = 1000
