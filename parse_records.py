import tensorflow as tf 
import os

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
TFRECORD_FOLDER_NAME = os.getenv("TFRECORD_FOLDER_NAME")
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
PRETRAIN_NUM_FILES = int(os.getenv("PRETRAIN_NUM_FILES"))

def parse_tfr_element(element, MODEL_MAX_LENGTH = MODEL_MAX_LENGTH, MAX_SUMMARY_LENGTH = MAX_SUMMARY_LENGTH):
    """Parse TFRecords element, structure same as serializing
    Args:
        element: TFRecords row
    Outputs: Dictionary of data for one article
    """
    data = {
    'input_ids': tf.io.FixedLenFeature([], tf.string),
    'attention_mask': tf.io.FixedLenFeature([], tf.string),
    #'decoder_input_ids': tf.io.FixedLenFeature([], tf.string),
    'labels': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    input_ids = content['input_ids']
    attention_mask = content['attention_mask']
    #decoder_input_ids = content['decoder_input_ids']
    labels = content['labels']
    #get our feature and convert str to tensor
    feature1 = tf.squeeze(tf.reshape(tf.io.decode_raw(input_ids, out_type=tf.int32) ,shape=(1,MODEL_MAX_LENGTH+3))[:,3:])
    feature2 = tf.squeeze(tf.reshape(tf.io.decode_raw(attention_mask, out_type=tf.int32) ,shape=[1,MODEL_MAX_LENGTH+3])[:,3:])
    #feature3 = tf.squeeze(tf.reshape(tf.io.decode_raw(decoder_input_ids, out_type=tf.int32) ,shape=[1,MAX_SUMMARY_LENGTH+3])[:,3:])
    #label = tf.ones(tf.shape(feature3),dtype=tf.int32)*-100 #no MLM
    label = tf.squeeze(tf.reshape(tf.io.decode_raw(labels, out_type=tf.int32) ,shape=[1,MAX_SUMMARY_LENGTH+3])[:,3:]) # uint16 barely enough for 64k vocabs
    return {'input_ids' :feature1,'attention_mask':feature2, 'labels':label} # 'decoder_input_ids':feature3


def parse_tfr_element_O(element, MODEL_MAX_LENGTH = MODEL_MAX_LENGTH, MAX_SUMMARY_LENGTH = MAX_SUMMARY_LENGTH):
    """Parse TFRecords element, structure same as serializing (for 64k data)
    Args:
        element: TFRecords row
    Outputs: Dictionary of data for one article
    """
    data = {
    'input_ids': tf.io.FixedLenFeature([], tf.string),
    'attention_mask': tf.io.FixedLenFeature([], tf.string),
    'decoder_input_ids': tf.io.FixedLenFeature([], tf.string),
    'labels': tf.io.FixedLenFeature([], tf.string)
    }

    content = tf.io.parse_single_example(element, data)

    input_ids = content['input_ids']
    attention_mask = content['attention_mask']
    decoder_input_ids = content['decoder_input_ids']
    labels = content['labels']
    #get our feature and convert str to tensor
    # idk why input_ids and decoder_input_ids always begin with 136 token, not ok
    padding = tf.constant([[0,1]])
    feature1 = tf.squeeze(tf.reshape(tf.io.decode_raw(input_ids, out_type=tf.int32) ,shape=(1,MODEL_MAX_LENGTH+3))[:,3:])
    feature2 = tf.squeeze(tf.reshape(tf.io.decode_raw(attention_mask, out_type=tf.int32) ,shape=[1,MODEL_MAX_LENGTH+3])[:,3:])
    feature3 = tf.pad(tf.squeeze(tf.reshape(tf.io.decode_raw(decoder_input_ids, out_type=tf.int32) ,shape=[1,MAX_SUMMARY_LENGTH+3])[:,4:]),padding,"CONSTANT")
    label = tf.squeeze(tf.reshape(tf.io.decode_raw(labels, out_type=tf.int32) ,shape=[1,MAX_SUMMARY_LENGTH+3])[:,3:]) # uint32 barely enough for 64k vocabs
    return {'input_ids' :feature1,'attention_mask':feature2,'decoder_input_ids':feature3, 'labels':label}

def get_dataset(tfr_dir: str = f"gs://{GCS_BUCKET_NAME}/records/{TFRECORD_FOLDER_NAME}",
                pattern: str = "{}_{}.tfrecord",
                mode = "pretrain",
                num_files = PRETRAIN_NUM_FILES,
                files = None):
    """Get dataset from list of tfrecord filename
    Args:
        tfr_dir: path to directory tfrecords in google storage bucket
        pattern: string format for tfrecord filename
        num_files: number of files, used in filename pattern
    Output:
        tf.dataset from tfrecord files"""
    if files == None:

        if num_files == None:
            if mode == "pretrain":
                num_files = PRETRAIN_NUM_FILES
            elif mode == "finetune":
                num_files = FINETUNE_NUM_FILES

        files = [os.path.join(tfr_dir, pattern.format(mode,idx)) for idx in range(num_files)]
    
    # create the dataset
    dataset = tf.data.TFRecordDataset(files)
    # pass every single feature through our mapping function
    dataset = dataset.map(parse_tfr_element)
    return dataset

def get_dataset_partitions_tf(dataset, BATCH_SIZE, val_size = 320000, shuffle_size=10000):
    dataset = dataset.shuffle(shuffle_size*100, reshuffle_each_iteration=False)
    val_dataset = dataset.take(val_size).shuffle(shuffle_size).batch(BATCH_SIZE, drop_remainder=True).shuffle(shuffle_size) 
    train_dataset = dataset.skip(val_size).shuffle(shuffle_size).batch(BATCH_SIZE, drop_remainder=True).shuffle(shuffle_size)
    return train_dataset, val_dataset