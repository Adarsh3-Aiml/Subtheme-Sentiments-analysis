import torch
from transformers import BertTokenizer


model = "base" # bert model type
INPUT_FILE = "input/Evaluation-dataset.csv" # path of input file
MAX_LEN = 128 # maximum lenght of token for bert tokenizer
PRE_TRAINED_MODEL = "bert-base-uncased" if model =='base' else "bert-large-uncased" # bert pretrained nodel
NUM_LABELS = 23 # Numbers of labels to predict
MODEL_PATH = "output/bert_model.pth.tar" # save model path
ENCODER_PATH = "output/encoder.pkl" # save label encoder path
LEARNING_RATE = 1e-04 if model == "base" else 1e-05 # learning rate training
EPOCHS = 5 # no. of epochs for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # device support for training
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL) # tokenizer for converting text to numercial values