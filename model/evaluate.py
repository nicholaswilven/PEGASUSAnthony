import tensorflow as tf
tf.random.set_seed(42)
print("Tensorflow version " + tf.__version__)
import os
import json
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer

from transformers import TFPegasusForConditionalGeneration, PegasusTokenizerFast

from utils.parse_records import get_dataset
from utils.model_config import get_config
from utils.sentencepiece_tokenizer import fetch_tokenizer

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MIN_SUMMARY_LENGTH = int(os.getenv("MIN_SUMMARY_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
LOAD_CKPT_PATH = os.getenv("LOAD_CKPT_PATH")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE"))
repo_name = os.getenv("REPO_NAME")

num_beams = 4

# Use TPU!
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

AUTO = tf.data.experimental.AUTOTUNE
tfr_dir = f"gs://{GCS_BUCKET_NAME}"+"/records/{}/{}_{}.tfrecord"

# Parse sys args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", help = "dataset name", default = "indosum_32k_test", type = str)
parser.add_argument("--num_files", help = "num files", default = 1, type = int)
args = parser.parse_args()

f = [tfr_dir.format(args.dataset_name,"finetune",idx) for idx in range(args.num_files)]

dataset = get_dataset(files = f).prefetch(AUTO)
dataset_np = list(dataset.as_numpy_iterator())

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'])
with tpu_strategy.scope():
    if repo_name:
        model = TFPegasusForConditionalGeneration.from_pretrained(repo_name)
        tokenizer = PegasusTokenizerFast.from_pretrained(repo_name)
    else:
        model = TFPegasusForConditionalGeneration(get_config(VOCAB_SIZE))
        model.load_weights(LOAD_CKPT_PATH)
        tokenizer = fetch_tokenizer()

print(f'{args.dataset_name}: Generating {len(dataset_np)} samples.')
sep = '|'
print(f'article{sep}gold{sep}summary{sep}rouge1{sep}rouge2{sep}rougeL')

batch_size = 16
from math import ceil
num_batch = ceil(len(dataset_np)/batch_size)
for idx in range(num_batch):
    row = dataset_np[idx*batch_size:(idx+1)*batch_size]
    input_ids = [r['input_ids'] for r in row]
    labels = [r['labels'] for r in row]
    input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    label = tokenizer.batch_decode(labels, skip_special_tokens=True)
    with tpu_strategy.scope():
        x = model.generate(input_ids = tf.convert_to_tensor(input_ids,dtype=tf.int32),
                            min_new_tokens = MIN_SUMMARY_LENGTH,
                            max_new_tokens = MAX_SUMMARY_LENGTH,
                            # early_stopping  = True, # want to explore full potential
                            num_beams = num_beams,
                            num_return_sequences = 1,
                            no_repeat_ngram_size = 1,
                            temperature = 0.7,
                            top_p = 0.75
                            #repetition_penalty = 2, # no need because no repeatings unigrams!
                            #encoder_repetition_penalty = 2,
                            #diversity_penalty = 0.1
                            )
    result = tokenizer.batch_decode(x, skip_special_tokens=True)
    for i in range(batch_size):
        res = result[i]
        score1 = scorer.score(res, label[i])['rouge1'].fmeasure
        score2 = scorer.score(res, label[i])['rouge2'].fmeasure
        scoreL = scorer.score(res, label[i])['rougeL'].fmeasure
        print(f'{input_text[i]}{sep}{label[i]}{sep}{res}{sep}{score1}{sep}{score2}{sep}{scoreL}')
    