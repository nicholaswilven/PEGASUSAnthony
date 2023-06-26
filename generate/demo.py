import tensorflow as tf
tf.random.set_seed(42)
print("Tensorflow version " + tf.__version__)
import os
import json
import pandas as pd
from rouge_score import rouge_scorer

from transformers import TFPegasusForConditionalGeneration, PegasusTokenizerFast

from utils.cleaning import process_input_eval
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

repo_name = "thonyyy/pegasus-indonesian-base_finetune"
num_beams = 4

# Use TPU!
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

AUTO = tf.data.experimental.AUTOTUNE
tfr_dir = f"gs://{GCS_BUCKET_NAME}"+"/records/{}/{}_{}.tfrecord"

ds = {}
num_files["indosum_32k_test"] = 1
num_files["liputan6_32k_test"] = 1
num_files["xlsum_32k_test"] = 1
num_files["indosum_32k"] = 4
num_files["liputan6_32k"] = 11
num_files["xlsum_32k"] = 2

scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'])
with tpu_strategy.scope():
    if repo_name:
        model = TFPegasusForConditionalGeneration.from_pretrained(repo_name)
        tokenizer = PegasusTokenizerFast.from_pretrained(repo_name)
    else:
        model = TFPegasusForConditionalGeneration(get_config(VOCAB_SIZE))
        model.load_weights(LOAD_CKPT_PATH)
        tokenizer = fetch_tokenizer()

def generate_from_index(dataset_name , idx , num_beams : int = num_beams):
    if dataset_name not in ds.keys():
        f = [tfr_dir.format(dataset_name,"finetune",idx) for idx in range(num_files[dataset_name])]
        dataset = get_dataset(files = f).prefetch(AUTO)
        ds[dataset_name] = list(dataset.as_numpy_iterator())
    
    row = ds[dataset_name][idx]
    input_text = tokenizer.decode(row['input_ids'], skip_special_tokens=True)
    label = tokenizer.decode(row['labels'], skip_special_tokens=True)
    with tpu_strategy.scope():
        x = model.generate(row['input_ids'],
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

    result = tokenizer.batch_decode(x, skip_special_tokens=True)
    
    respond = {}
    respond['article'] = input_text
    respond['gold'] = label
    summary_list = []
    for summary in result:
        obj = {}
        obj['summary'] = summary
        for metric in ['rouge1','rouge2','rougeL']:
            obj[metric] = scorer.score(summary, label)[metric].fmeasure
        summary_list.append(obj)
    respond['result'] = summary_list
    return respond
    
def abs_summary(input_text : str = "sampel artikel.", num_beams:int = num_beams):
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
    
    result = tokenizer.batch_decode(x, skip_special_tokens=True)
    
    respond = {}
    respond['article'] = input_text
    summary_list = []
    for summary in result:
        obj = {}
        obj['summary'] = summary
        summary_list.append(obj)
    respond['result'] = summary_list
    return respond