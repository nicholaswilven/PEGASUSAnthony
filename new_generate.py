#from __future__ import absolute_import
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)
print("Tensorflow version " + tf.__version__)
import os
import json

from cleaning import text_cleaning
from parse_records import get_dataset, get_dataset_partitions_tf
from transformers import TFPegasusForConditionalGeneration, AutoTokenizer
from model import get_config

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FINETUNE_DATA_LIST = json.loads(os.environ['FINETUNE_DATA_LIST'])

tokenizer = AutoTokenizer.from_pretrained("google/pegasus-large")

def process_input_eval(text, tokenizer = tokenizer):
    t = text_cleaning(text)
    return tokenizer(t,
              return_tensors = "tf",
              padding = 'max_length',
              max_length = MODEL_MAX_LENGTH,
              truncation = True
              )

# Parse sys args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--vocab_size", help = "vocab size model and tokenizer", default = 16103, type=int)
parser.add_argument("--load_ckpt_path", help = "path toload model weights", type=str)
args = parser.parse_args()

# Hardcode for banning . in generation 
if args.vocab_size == 32103:
    dot_ids = None # Update according to tokenizer
else:
    dot_ids = None

try:
    with open('article.txt') as f:
        input_text = f.read()
except:
    print('File doesn\'t exist. Pulling article from validation dataset')
    input_text = None

if input_text == None: # Take random article from finetune dataset if not specified
    import pandas as pd
    import random
    prefix_dir: str = f"gs://{GCS_BUCKET_NAME}/data",
    filelist = [os.path.join(prefix_dir, f) for f in FINETUNE_DATA_LIST]
    df = pd.read_parquet(filelist(random.randint(0,1)))
    n = len(df)
    idx = random.randint(0,n-1)
    input_text = df.loc[idx,'input']
    print("Article:", input_text)
    print("gold:", df.loc[idx,'labels'])
else:
    print("Article:", input_text)

from googletrans import Translator
translator = Translator()
input_text_trs = translator.translate(input_text, dest='en', src='id').text

t = process_input_eval(input_text_trs)

# Use TPU!
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

with tpu_strategy.scope():
    model = TFPegasusForConditionalGeneration.from_pretrained('google/pegasus-large')
    x = model.generate(**t, max_new_tokens = MAX_SUMMARY_LENGTH,
                        # early_stopping  = True, # want to explore full potential
                        num_beams = 8,
                        #repetition_penalty = 2, # no need because no repeatings unigrams!
                        no_repeat_ngram_size = 2,
                        temperature = 0.7,
                        top_p = 0.75,
                        encoder_repetition_penalty = 2,
                        diversity_penalty = 0.1,
                        num_return_sequences = 8,
                        bad_words_ids = dot_ids
                        )

summary = tokenizer.batch_decode(x, skip_special_tokens=True)
print("Summaries:")
for sum in summary:
    output_summary = translator.translate(sum, dest='id', src='en').text
    print(output_summary)
