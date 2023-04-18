import tensorflow as tf 
import os

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))

def parse_tfr_element(element, MODEL_MAX_LENGTH = MODEL_MAX_LENGTH, MAX_SUMMARY_LENGTH = MAX_SUMMARY_LENGTH):
    """Parse TFRecords element, structure same as serializing
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
    feature1 = tf.reshape(tf.io.decode_raw(input_ids, out_type=tf.int32) ,shape=(1,MODEL_MAX_LENGTH+3))[:,3:]
    print(tf.shape(feature1))
    
    feature2 = tf.reshape(tf.io.decode_raw(attention_mask, out_type=tf.int32) ,shape=[1,MODEL_MAX_LENGTH+3])[:,3:]
    feature3 = tf.reshape(tf.io.decode_raw(decoder_input_ids, out_type=tf.int32) ,shape=[1,MAX_SUMMARY_LENGTH+3])[:,3:]
    label = tf.reshape(tf.io.decode_raw(labels, out_type=tf.int32) ,shape=[1,MAX_SUMMARY_LENGTH+3])[:,3:] # uint32 barely enough for 64k vocabs
    return {'input_ids' :feature1,'attention_mask':feature2,'decoder_input_ids':feature3, 'labels':label}

def get_dataset(tfr_dir: str = f"gs://{GCS_BUCKET_NAME}/records/fulldata",
                pattern: str = "{}_{}.tfrecord",
                mode = "pretrain",
                num_files = 276):
    """Get dataset from list of tfrecord filename
    Args:
        tfr_dir: path to directory tfrecords in google storage bucket
        pattern: string format for tfrecord filename
        num_files: number of files, used in filename pattern
    Output:
        tf.dataset from tfrecord files"""
        
    files = [os.path.join(tfr_dir, pattern.format(mode,idx)) for idx in range(num_files)]
    # create the dataset
    dataset = tf.data.TFRecordDataset(files)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset

def get_dataset_partitions_tf(dataset, BATCH_SIZE, val_size = 320000, shuffle_size=1000):
    dataset = dataset.shuffle(shuffle_size, seed=12, reshuffle_each_iteration=False)
    val_dataset = dataset.take(val_size)    
    train_dataset = dataset.skip(val_size)
    tf_batch_shuffle = lambda ds : ds.shuffle(shuffle_size, seed=12).batch(BATCH_SIZE, drop_remainder=True).shuffle(shuffle_size, seed=12)
    return tf_batch_shuffle(train_dataset), tf_batch_shuffle(val_dataset)