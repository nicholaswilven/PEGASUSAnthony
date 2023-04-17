from convert_to_records import convert_parquet_to_records
import tensorflow as tf

'''
# Model specific parameters
tf.flags.DEFINE_string("data_dir", "",
                       "Path to directory containing the MNIST dataset")
tf.flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")

tf.flags.DEFINE_bool("enable_predict", True, "Do some predictions at the end")
tf.flags.DEFINE_integer("iterations", 50,
                        "Number of iterations per TPU training loop.")

FLAGS = tf.flags.FLAGS

def train_input_fn(params):
  """train_input_fn defines the input pipeline used for training."""
  batch_size = params["batch_size"]
  data_dir = params["data_dir"]
  # Retrieves the batch size for the current shard. The # of shards is
  # computed according to the input pipeline deployment. See
  # `tf.contrib.tpu.RunConfig` for details.
  ds = dataset.train(data_dir).cache().repeat().shuffle(
      buffer_size=50000).apply(
          tf.contrib.data.batch_and_drop_remainder(batch_size))
  images, labels = ds.make_one_shot_iterator().get_next()
  return images, labels
'''

if __name__=="__main__":
    #FLAGS
    convert_parquet_to_records(mode='finetune',num_file = 521)