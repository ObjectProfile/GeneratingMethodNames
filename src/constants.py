from pathlib import Path
import seaborn as sns

WRITE_LOGS_TO_GOOGLE_DRIVE = False
DRIVE_DIR = 'NameGen'

DEVICE = 'cpu'

SOS_TOKEN = 0
EOS_TOKEN = 1
UNK_TOKEN = 2
PAD_TOKEN = 3

MAX_LENGTH = 50
DROP_DUPLICATES = True

TRAIN_PROP = 0.7
VALID_PROP = 0.1
TEST_PROP = 0.2

HIDDEN_SIZE = 256
LEARNING_RATE = 0.01
TEACHER_FORCING_RATIO = 0.5

NUM_ITER = 100000
LOG_EVERY = 1000

# Colors used in visualizations
COLOR_GREEN = sns.xkcd_rgb["faded green"]
COLOR_RED = sns.xkcd_rgb["pale red"]
COLOR_BLUE = sns.xkcd_rgb["medium blue"]
COLOR_YELLOW = sns.xkcd_rgb["ochre"]

# Parent to all other diretories
BASE_DIR = Path('/Volumes/MyPassport/Data/NameGen_test/')

DATASET_FILE          = str(BASE_DIR / 'Data' / 'methods_tokenized.csv')
TRAIN_PAIRS_FILE      = str(BASE_DIR / 'State' / 'Train-Valid-Test' / 'train_pairs.pkl')
VALID_PAIRS_FILE      = str(BASE_DIR / 'State' / 'Train-Valid-Test' / 'valid_pairs.pkl')
TEST_PAIRS_FILE       = str(BASE_DIR / 'State' / 'Train-Valid-Test' / 'test_pairs.pkl')
INPUT_LANG_FILE       = str(BASE_DIR / 'State' / 'Langs' / 'input_lang.pkl')
OUTPUT_LANG_FILE      = str(BASE_DIR / 'State' / 'Langs' / 'output_lang.pkl')
ITERS_COMPLETED_FILE  = str(BASE_DIR / 'State' / 'iters_completed.txt')
TRAINED_MODEL_FILE    = str(BASE_DIR / 'Models' / 'trained_model.pt')
LOG_FILE              = str(BASE_DIR / 'Logs' / 'log.txt')
TRAIN_LOG_FILE        = str(BASE_DIR / 'Logs' / 'train-log.txt')
VALIDATION_NAMES_FILE = str(BASE_DIR / 'Results' / 'valid_names.csv')
TEST_NAMES_FILE       = str(BASE_DIR / 'Results' / 'test_names.csv')
HISTORIES_FILE        = str(BASE_DIR / 'Results' / 'histories.csv')
LOSS_IMG_FILE         = str(BASE_DIR / 'Img' / 'loss.png')
BLEU_IMG_FILE         = str(BASE_DIR / 'Img' / 'bleu.png')
ROUGE_IMG_FILE        = str(BASE_DIR / 'Img' / 'rouge.png')
F1_IMG_FILE           = str(BASE_DIR / 'Img' / 'f1.png')
ALL_SCORES_IMG_FILE   = str(BASE_DIR / 'Img' / 'all_scores.png')
NUM_NAMES_IMG_FILE    = str(BASE_DIR / 'Img' / 'num_names.png')
