#from __future__ import absolute_import
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
import os
import re
import unicodedata

from parse_records import get_dataset, get_dataset_partitions_tf
from transformers import TFPegasusForConditionalGeneration
from model import get_config
from sentencepiece_tokenizer import fetch_tokenizer

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
FINETUNE_DATA_LIST = json.loads(os.environ['FINETUNE_DATA_LIST'])

def remove_news_headline(text,delim):
    x = text.split(delim)
    if len(x)>1: # buang yang bukan konten
        return " ".join(x[1:])
    else:
        return x[0]

def text_cleaning(input_string, is_news = True):
    lowercase = input_string.lower()
    # stripped_html = BeautifulSoup(lowercase, 'html.parser').get_text()
    remove_link = re.sub(r'(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)', '', lowercase).replace("&amp;","&")
    remove_bullet = "\n".join([T for T in remove_link.split('\n') if '•' not in T and "baca juga:" not in T])
    remove_accented = unicodedata.normalize('NFKD', remove_bullet).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # remove news headline
    if is_news:
        y = remove_news_headline(remove_accented,'- ')
        y = remove_news_headline(y,'– ') 
    else:
        y = remove_accented
    remove_parentheses = re.sub("([\(\|]).*?([\)\|])", "\g<1>\g<2>", y) 
    remove_punc = re.sub(r"[^\w\d.\s]+",' ', remove_parentheses)
    remove_num_dot = re.sub(r"(?<=\d)\.|\.(?=\d)|(?<=#)\.","",remove_punc)
    remove_extra_whitespace =  re.sub(r'^\s*|\s\s*', ' ', remove_num_dot).strip()
    return ".".join([s for s in remove_extra_whitespace.strip().split('.') if len(s.strip())>10]).replace("_","")

tokenizer = fetch_tokenizer()
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
parser.add_argument("--input_text", help = "text to summarize", type=str)
args = parser.parse_args()

# Use TPU!
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

# This one is for tensorboard
tf.profiler.experimental.server.start(6000)

# Hardcode for banning . in generation 
if args.vocab_size == 16103:
    dot_ids = [[128], [106]]
elif args.vocab_size == 32103:
    dot_ids == [[127],[106]]

input_text = args.input_text
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
    
with tpu_strategy.scope():
    model = TFPegasusForConditionalGeneration(get_config(args.vocab_size))
    model.load_weights(args.load_ckpt_path)
    x = model.generate(**t, max_new_tokens = MAX_SUMMARY_LENGTH,
                        # early_stopping  = True, # want to explore full potential
                        num_beams = 8,
                        #repetition_penalty = 2, # no need because no repeatings unigrams!
                        no_repeat_ngrams_size = 1,
                        temperature = 0.7,
                        top_p = 0.75,
                        encoder_repetition_penalty = 2
                        diversity_penalty = 0.1,
                        num_return_sequences = 8,
                        bad_words_ids = dot_ids
                        )

summary = token.batch_decode(x, skip_special_tokens=True)
print("Summaries:")
for sum in summary:
    print(summary)