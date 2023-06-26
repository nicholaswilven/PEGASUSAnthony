import tensorflow as tf
tf.random.set_seed(42)
import os
import nltk
from math import ceil
import pandas as pd
import json
from datasets import load_dataset

from .sentencepiece_tokenizer import _tokenize_inputs, fetch_tokenizer
from .gap_sentence_generation import _E_GSG
from .cleaning import cleaning_oscar

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
SAMPLE_PER_FILE = int(os.getenv("SAMPLE_PER_FILE"))

tokenizer = fetch_tokenizer()

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_pretrain(feature1, feature2, feature4):
    """Python function for serializing tensor -> string -> byte_list -> features -> string
    Args:
        feature1: input_ids tensor with shape (sample_size, MODEL_MAX_LENGTH)
        feature2: attention_mask tensor with shape (sample_size, MODEL_MAX_LENGTH)
        feature4: labels tensor with shape (sample_size, MAX_SUMMARY_LENGTH)
    Outputs: serialized string for train.example
    """

    feature = {
        'input_ids': _bytes_feature(tf.io.serialize_tensor(feature1)),
        'attention_mask': _bytes_feature(tf.io.serialize_tensor(feature2)),
        'labels': _bytes_feature(tf.io.serialize_tensor(feature4))
    }

    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def tf_serialize(f0,f1,f3):
    """ Tensorflow wrapper for Python serialize_pretrain function 
    Args: refer to serialize_pretrain
    Outputs: serialized tf.scalar for train.example
    """
    tf_string = tf.py_function(
        serialize_pretrain,
        (f0, f1, f3),  # Pass these args to the above function.
        tf.string)      # The return type is `tf.string`.
    return tf.reshape(tf_string, ()) # The result is a scalar.

def serialize_examples(dataset, mode = "pretrain"):
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
    tf_dataset = tf.data.Dataset.from_tensor_slices((
        tokenized_df['input_ids'],
        tokenized_df['attention_mask'],
        tokenized_df['labels']
        ))
    serialized_tf_dataset = tf_dataset.map(tf_serialize)
    return serialized_tf_dataset
        
def convert_df_to_records(df,
                          TFRecord_folder_name : str, 
                          mode : str = "pretrain",                               
                          sample_per_file : int = SAMPLE_PER_FILE,
                          out_dir : str = f"gs://{GCS_BUCKET_NAME}/records/",
                          start_from : int = 0):
    """ Runnable python function to convert parquet file to tfrecord files 
    Args:
        mode = 'pretrain' or 'finetune'. Specify which dataset to process
        num_files = number of tfrecord files generated, each ~100MB
        prefix_dir = path to parquet files (slightly hardcoded)
        out_dir = path for output TFRecord folder
        start_from = helper params to continue if process terminated 
    """
    print(f"Total records : {len(df)}")
    num_files = ceil(len(df)/sample_per_file)
    for idx in range(start_from,num_files):
        print(f"Printing file {idx} from {num_files}")
        # Writer object
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir,TFRecord_folder_name,f'{mode}_{idx}.tfrecord'))
        # Process + serialize the dataframe using wraper function
        temp_df = df.iloc[idx*sample_per_file:(idx+1)*sample_per_file].to_dict(orient="list")
        serialized_tf_dataset = serialize_examples(temp_df, mode = mode)
        # Write the file
        writer.write(serialized_tf_dataset)     
        print(f"Printing file {idx} DONE!")  

def convert_ds_to_records(dataset,
                          TFRecord_folder_name : str,
                          mode : str = "finetune",
                          sample_per_file : int = SAMPLE_PER_FILE,
                          out_dir : str = f"gs://{GCS_BUCKET_NAME}/records/{TFRECORD_FOLDER_NAME}",
                          start_from : int = 0,
                          num_proc : int = 96):
    """ Runnable python function to convert parquet file to tfrecord files via hf datasets, benefits from multiprocessing. Only for pretrain dataset with columns ['text']
    Args:
        out_dir = path for output TFRecord folder
        prefix_dir = path to parquet files (slightly hardcoded)
        sample_per_file = number of datapoints in one TFRecord file, each ~100MB
        start_from = helper params to continue if process terminated 
        num_proc = number of CPU cores/ workers available
    """
    # Load parquet file as datasets, combine into one
    if mode == "pretrain":
        # Apply gap sentence generation
        tokenizer = fetch_tokenizer()
        dataset = dataset.map(_E_GSG,
                              batched = True,
                              remove_columns = ["text"],
                              num_proc = num_proc)
    # Tokenize dataset
    tokenized_dataset = dataset.map(lambda x : _tokenize_inputs(x, tokenizer = tokenizer),
                                              batched = True,
                                              remove_columns = ["input"],
                                              num_proc = num_proc,
                                              load_from_cache_file = False)

    print(f"Total records : {len(tokenized_dataset)}")
    num_files = ceil(len(tokenized_dataset)/sample_per_file)
    for idx in range(start_from,num_files):
        print(f"Printing file {idx} from {num_files}")
        # Writer object
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir,TFRecord_folder_name,f'pretrain_{idx}.tfrecord'))
        # Convert dataset to tf.dataset
        temp_ds = tokenized_dataset.shard(num_shards=num_files,index = idx)
        tf_dataset = tf.data.Dataset.from_tensor_slices((
                temp_ds['input_ids'],
                temp_ds['attention_mask'],
                temp_ds['labels']
                ))
        serialized_tf_dataset = tf_dataset.map(tf_serialize)
        # Write the file
        writer.write(serialized_tf_dataset)     
        print(f"Printing file {idx} DONE!") 

def convert_oscar_to_records(TFRecord_folder_name : str,
                               sample_per_file : int = SAMPLE_PER_FILE,
                               out_dir : str = f"gs://{GCS_BUCKET_NAME}/records/",   
                               start_from : int = 0,
                               num_proc : int = 96):
                            
    """ Runnable python function to convert OSCAR corpus dataset to tfrecord files as pretrain dataset, benefits from multiprocessing.
    Args:
        out_dir = path for output TFRecord folder
        sample_per_file = number of datapoints in one TFRecord file, each ~100MB
        start_from = helper params to continue if process terminated 
        num_proc = number of CPU cores/ workers available
    """
    dataset = load_dataset("oscar-corpus/OSCAR-2201",
                           use_auth_token = True,
                           language = "id",
                           split = "train")
    # Cleaning the dataset
    clean_dataset = dataset.map(cleaning_oscar,
                                batched = True,
                                remove_columns = ["id","meta"],
                                num_proc = num_proc,
                                load_from_cache_file = False)
    # Apply gap sentence generation and tokenizing
    processed_dataset = clean_dataset.map(_E_GSG,
                                          batched = True,
                                          remove_columns = ["text"],
                                          num_proc = num_proc,
                                          load_from_cache_file = False)
    processed_dataset = processed_dataset.filter(lambda example : example['input'] != "SKIP",
                                                 num_proc = num_proc,
                                                 load_from_cache_file=  False)
    # Tokenize dataset
    tokenizer = fetch_tokenizer()
    tokenized_dataset = processed_dataset.map(lambda x : _tokenize_inputs(x, tokenizer = tokenizer),
                                              batched = True,
                                              remove_columns = ["input"],
                                              num_proc = 96,
                                              load_from_cache_file = False)
    print(f"Total records : {len(tokenized_dataset)}")
    num_files = ceil(len(tokenized_dataset)/sample_per_file)
    for idx in range(start_from,num_files):
        print(f"Printing file {idx} from {num_files}")
        writer = tf.data.experimental.TFRecordWriter(os.path.join(out_dir,TFRecord_folder_name,f'pretrain_{idx}.tfrecord'))
        temp_ds = tokenized_dataset.shard(num_shards=num_files,index = idx)
        tf_dataset = tf.data.Dataset.from_tensor_slices((
                temp_ds['input_ids'],
                temp_ds['attention_mask'],
                temp_ds['labels']
                ))
        serialized_tf_dataset = tf_dataset.map(tf_serialize)
        writer.write(serialized_tf_dataset)     
        print(f"Printing file {idx} DONE!")

