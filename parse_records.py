import tensorflow as tf 
import os

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MODEL_SUMMARY_LENGTH"))

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
                pattern: str = "pretrain_{}.tfrecord",
                num_files = 276):
    """Get dataset from list of tfrecord filename
    Args:
        tfr_dir: path to directory tfrecords in google storage bucket
        pattern: string format for tfrecord filename
        num_files: number of files, used in filename pattern
    Output:
        tf.dataset from tfrecord files"""
        
    files = [os.path.join(tfr_dir, pattern.format(idx)) for idx in range(num_files)]
    # create the dataset
    dataset = tf.data.TFRecordDataset(files)

    # pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset