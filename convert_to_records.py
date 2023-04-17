import os
import nltk
from math import ceil
import pandas as pd
import tensorflow as tf

from sentencepiece_tokenizer import _tokenize_inputs, fetch_tokenizer
from gap_sentence_generation import _E_GSG

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_pretrain(feature1, feature2, feature3, feature4):
    """Python function for serializing tensor -> string -> byte_list -> features -> string
    Args:
        feature1: input_ids tensor with shape (sample_size, MODEL_MAX_LENGTH)
        feature2: attention_mask tensor with shape (sample_size, MODEL_MAX_LENGTH)
        feature3: decoder_input_ids tensor with shape (sample_size, MAX_SUMMARY_LENGTH)
        feature4: labels tensor with shape (sample_size, MAX_SUMMARY_LENGTH)
    Outputs: serialized string for train.example
    """

    feature = {
        'input_ids': _bytes_feature(tf.io.serialize_tensor(feature1)),
        'attention_mask': _bytes_feature(tf.io.serialize_tensor(feature2)),
        'decoder_input_ids': _bytes_feature(tf.io.serialize_tensor(feature3)),
        'labels': _bytes_feature(tf.io.serialize_tensor(feature4))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize(f0,f1,f2,f3):
    """ Tensorflow wrapper for Python serialize_pretrain function 
    Args: refer to serialize_pretrain
    Outputs: serialized tf.scalar for train.example
    """
    tf_string = tf.py_function(
        serialize_pretrain,
        (f0, f1, f2, f3),  # Pass these args to the above function.
        tf.string)      # The return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar.

def serialize_examples(dataset, tokenizer = fetch_tokenizer(), mode = "pretrain"):
    """ Convert dataframe/dict to serialized tf.dataset
    Args: 
        dataset: dataframe/dict with keys ['input'] for mode = 'pretrain' 
                    and ['input,'labels'] for mode = 'finetune'
        tokenizer: PegasusTokenizer object used
        mode = 'pretrain' or 'finetune'. Pretrain dataset have gap sentence generation processing
    """
    if mode == "pretrain":
        dataset = _E_GSG(
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
        
def convert_parquet_to_records(mode = "pretrain",
                                num_file = 276 ,
                                prefix_dir: str = "data/pegasusanthony_fix",
                                out_dir: str = f"gs://{GCS_BUCKET_NAME}/records/fulldata"
                                ):
    """ Runnable python function to convert parquet file to tfrecord files 
    Args:
        mode = 'pretrain' or 'finetune'. Specify which dataset to process
        num_file = number of tfrecord files generated (each ~100MB)
        prefix_dir = path to parquet files (slightly hardcoded)
        outdir_dir = path to tfrecord files
    """
    if mode == "pretrain":
        filenames = [os.path.join(prefix_dir, f) 
                     for f in ['ccnews-id.parquet.gzip',
                               'id-wiki-clean.parquet',
                               'news-2017-clean.parquet']
                               ]
    elif mode == "finetune":
        filenames = [os.path.join(prefix_dir, f) 
                     for f in ["reddit-tldr.parquet.gzip",
                               "liputan6.parquet",
                               "gigaword.parquet",
                               "indosum.parquet"]
                               ]
    else:
        raise ValueError('Please specify the mode, "pretrain" or "finetune"')
    
    print(f"Running {mode} mode")
    
    data = []
    for filename in filenames:
        df = pd.read_parquet(filename)
        if filename == os.path.join(prefix_dir,"ccnews-id.parquet.gzip"):
            df['sent'] = df['text'].apply(lambda x: len(nltk.tokenize.sent_tokenize(x)))
            df = df[df['sent']>2]
            df = df[['text']]
            print("Preprocessed CC News!")
        elif filename == os.path.join(prefix_dir,"gigaword.parquet"):
            df.rename(columns={'summary':'labels','document':'input'},inplace=True)
        data.append(df)
        print(f"Read {filename} done")
        print(df.head())
        del df
    df = pd.concat(data).reset_index(drop=True)
    del data
    print(f"Total records : {len(df)}")
    sample_per_file = ceil(len(df)/num_file)
    for idx in range(273,num_file):
        print(f"Printing file {idx} from {num_file}")
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir,f'{mode}_{idx}.tfrecord'))
        temp_df = df.iloc[idx*sample_per_file:(idx+1)*sample_per_file].to_dict(orient="list")
        serialized_tf_dataset = serialize_examples(temp_df, mode = mode)
        writer.write(serialized_tf_dataset)       

