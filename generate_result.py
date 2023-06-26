import tensorflow as tf
tf.random.set_seed(42)
print("Tensorflow version " + tf.__version__)
import os
import json
import pandas as pd
from rouge_score import rouge_scorer

from cleaning import process_input_eval
from parse_records import get_dataset, get_dataset_partitions_tf
from transformers import TFPegasusForConditionalGeneration
from model import get_config
from sentencepiece_tokenizer import fetch_tokenizer

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MIN_SUMMARY_LENGTH = int(os.getenv("MIN_SUMMARY_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
LOAD_CKPT_PATH = os.getenv("LOAD_CKPT_PATH")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE"))

# Use TPU!
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

prefix_dir = f"gs://{GCS_BUCKET_NAME}/data_new"
filelist = [pd.read_parquet(os.path.join(prefix_dir, f)) for f in FINETUNE_DATA_LIST]
df = pd.concat(filelist)
tokenizer = fetch_tokenizer()

with tpu_strategy.scope():
    model = TFPegasusForConditionalGeneration(get_config(VOCAB_SIZE))
    model.build(input_shape = {"input_ids":[128, 512],"decoder_input_ids":[128,256]})
    model.load_weights(LOAD_CKPT_PATH)

scorer = rouge_scorer.RougeScorer(['rouge1'])

for idx in range(14000,len(df)):
    input_text = df.iloc[idx]['input']
    t = process_input_eval(input_text)
    with tpu_strategy.scope():
        x = model.generate(**t,
                            min_new_tokens = MIN_SUMMARY_LENGTH,
                            max_new_tokens = MAX_SUMMARY_LENGTH,
                            # early_stopping  = True, # want to explore full potential
                            num_beams = num_beams,
                            num_return_sequences = num_beams,
                            no_repeat_ngram_size = 1,
                            temperature = 0.7,
                            top_p = 0.75
                            #repetition_penalty = 2, # no need because no repeatings unigrams!
                            #encoder_repetition_penalty = 2,
                            #diversity_penalty = 0.1
                            )
    summary = tokenizer.batch_decode(x, skip_special_tokens=True)
    for sum in summary:
        s = scorer.score(sum, text_cleaning(df.iloc[idx]['labels']))['rouge1'].fmeasure
        if s > 0.2:
            print('index:',idx)
            print("Article:",input_text)
            print("gold:",df.iloc[idx]['labels'])
            print("summary:",sum)
            print("ROUGE 1 F1:",s)
            print('------------')