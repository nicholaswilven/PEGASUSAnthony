import os
import pandas as pd
import tensorflow as tf
from sentencepiece_tokenizer import _tokenize_inputs, init_tokenizer_fast, init_tokenizer_slow
from gap_sentence_generation import _e_gap_sentence_generation
import nltk
from math import ceil

import os
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
tokenizer_type = os.getenv("tokenizer_type")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_pretrain(feature1, feature2, feature3, feature4):
    feature = {
        'input_ids': _bytes_feature(tf.io.serialize_tensor(feature1)),
        'attention_mask': _bytes_feature(tf.io.serialize_tensor(feature2)),
        'decoder_input_ids': _bytes_feature(tf.io.serialize_tensor(feature3)),
        'labels': _bytes_feature(tf.io.serialize_tensor(feature4))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize(f0,f1,f2,f3):
    tf_string = tf.py_function(
        serialize_pretrain,
        (f0, f1, f2, f3),  # Pass these args to the above function.
        tf.string)      # The return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar.

def fetch_tokenizer(tokenizer_type = tokenizer_type):
    if tokenizer_type == "Unigram":
        tokenizer = init_tokenizer_fast()
    else:
        tokenizer = init_tokenizer_slow()
    return tokenizer

def serialize_examples(dataset, tokenizer = fetch_tokenizer(), mode = "pretrain"):
    if mode == "pretrain":
        dataset = _e_gap_sentence_generation(
            examples = dataset
            )
    tokenized_df = _tokenize_inputs(
        examples = dataset,
        tokenizer = tokenizer
        )
    del dataset
    tf_dataset = tf.data.Dataset.from_tensor_slices((
        tokenized_df['input_ids'],
        tokenized_df['attention_mask'],
        tokenized_df['decoder_input_ids'],
        tokenized_df['labels']
        ))
    del tokenized_df
    serialized_tf_dataset = tf_dataset.map(tf_serialize)
    del tf_dataset
    return serialized_tf_dataset
        
def convert_parquet_to_records(mode = "pretrain", num_file = 276 , prefix_dir: str = "data/pegasusanthony_fix", out_dir: str = f"gs://{GCS_BUCKET_NAME}/records/exp1"):
    if mode == "pretrain":
        filenames = [os.path.join(prefix_dir, f) 
                     for f in ['ccnews-id.parquet.gzip',
                               'id-wiki-clean.parquet',
                               'news-2017-clean.parquet']
                               ]
    elif mode == "finetune":
        filenames = [os.path.join(prefix_dir, f) 
                     for f in ["reddit-tldr.parquet.gzip",
                               "liputan6.parquet.gzip",
                               "gigaword.parquet.gzip",
                               "indosum.gzip"]
                               ]
    else:
        raise ValueError('Please specify the mode, "pretrain" or "finetune"')
    
    print(f"Running {mode} mode")
    
    data = []
    for filename in filenames:
        df = pd.read_parquet(filename)
        if filename == os.path.join(prefix_dir,"cccnews-id.parquet.gzip"):
            df['sent'] = df['text'].apply(lambda x: len(nltk.tokenize.sent_tokenize(x)))
            df = df[df['sent']>2]
            df = df[['text']]
            print("Preprocessed CC News!")
        data.append(df)
        print(f"Read {filename} done")
        del df
    df = pd.concat(data).reset_index(drop=True)
    del data
    print(f"Total records : {len(df)}")
    sample_per_file = ceil(len(df)/num_file)
    for idx in range(num_file):
        print(f"Printing file {idx} from {num_file}")
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir,f'{mode}_{idx}.tfrecord'))
        temp_df = df.iloc[idx*sample_per_file:(idx+1)*sample_per_file].to_dict(orient="list")
        serialized_tf_dataset = serialize_examples(temp_df, mode = mode)
        writer.write(serialized_tf_dataset)       

