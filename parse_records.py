import tensorflow as tf 
import glob
import os
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MODEL_SUMMARY_LENGTH"))

def parse_tfr_element(element, MODEL_MAX_LENGTH = MODEL_MAX_LENGTH, MAX_SUMMARY_LENGTH = MAX_SUMMARY_LENGTH):
    #use the same structure as above; it's kinda an outline of the structure we now want to create
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
    feature1 = tf.squeeze(tf.reshape(tf.io.decode_raw(input_ids, out_type=tf.int32) ,shape=(1,MODEL_MAX_LENGTH+3))[:,3:])
    print(tf.shape(feature1))
    
    feature2 = tf.squeeze(tf.reshape(tf.io.decode_raw(attention_mask, out_type=tf.int32) ,shape=[1,MODEL_MAX_LENGTH+3])[:,3:])
    feature3 = tf.squeeze(tf.reshape(tf.io.decode_raw(decoder_input_ids, out_type=tf.int32) ,shape=[1,MAX_SUMMARY_LENGTH+3])[:,3:])
    label = tf.squeeze(tf.reshape(tf.io.decode_raw(labels, out_type=tf.int32) ,shape=[1,MAX_SUMMARY_LENGTH+3])[:,3:]) # uint32 barely enough for 64k vocabs
    return {'input_ids' :feature1,'attention_mask':feature2,'decoder_input_ids':feature3, 'labels':label}

def get_dataset(tfr_dir: str = "/content/", pattern: str = "pretrain.tfrecord"):
    files = glob.glob(os.path.join(tfr_dir, pattern), recursive=False)
    print(files)

    #create the dataset
    dataset = tf.data.TFRecordDataset(files)

    #pass every single feature through our mapping function
    dataset = dataset.map(
        parse_tfr_element
    )

    return dataset