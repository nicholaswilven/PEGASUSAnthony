import tensorflow as tf
tf.random.set_seed(42)
print("Tensorflow version " + tf.__version__)
import os
import json
import pandas as pd
from rouge_score import rouge_scorer
from transformers import TFPegasusForConditionalGeneration

from utils.cleaning import process_input_eval, text_cleaning
from utils.parse_records import get_dataset, get_dataset_partitions_tf
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
    model.load_weights(LOAD_CKPT_PATH)

scorer = rouge_scorer.RougeScorer(['rouge1'])

def abs_summary(input_text : str = "sampel artikel.",
                idx : int = None,
                num_beams : int = 8):
    """ Sumamrize article (or from index dataset) using model specified in env. Called by FastAPI
    Args:
        input_text = article to summarize
        idx = index dataset to summarize
    Output:
        result: Dictionary for 
            input_text: article to summarize
            gold: human written summary, only for summary from index dataset
            summary_list:
                summary : model generated summary
                rouge1_f1 : ROUGE1 F1 score with gold, only for summary from index dataset
    """
    if idx != None:
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

    result = {}
    result['input_text'] = input_text
    if idx != None:
        gold = text_cleaning(df.iloc[idx]['labels'])
        result['gold'] = gold
        summary_list = []
        for sum in summary:
            obj = {}
            obj['rouge1_f1'] = scorer.score(sum, gold)['rouge1'].fmeasure
            obj['summary'] = sum
            summary_list.append(obj)
        result['result'] = summary_list
    else:
        summary_list = []
        for sum in summary:
            obj = {}
            obj['summary'] = sum
            summary_list.append(obj)
        result['result'] = summary_list
    return result