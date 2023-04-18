from parse_records import get_dataset, get_dataset_partitions_tf
from build_model import build_model
import tensorflow as tf
import os
import time
# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
BATCH_SIZE = os.getenv("BATCH_SIZE")
MAX_EPOCHS = os.getenv("MAX_EPOCHS")
PRETRAIN_LEARNING_RATE = os.getenv("PRETRAIN_LEARNING_RATE")

print('Load dataset!')
dataset = get_dataset(num_files = 2)
train_dataset, val_dataset = get_dataset_partitions_tf(dataset, BATCH_SIZE)

checkpoint_filepath = ('/home/hikari.netto.355/PEGASUSAnthony/checkpoints/weights-pretrain.ckpt')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only = True,
    monitor = "val_loss",
    mode = 'auto',
    verbose = 1,
    save_best_only = False
    )

def train_model():
    model = build_model()
    model.compile(optimizer = tf.keras.optimizers.experimental.Adafactor(learning_rate = PRETRAIN_LEARNING_RATE), metrics = ["accuracy"])
    print('Start training!')
    # Refer to https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit for more options
    model.fit(train_dataset,
        epochs = 1,
        verbose = 1,
        validation_data = val_dataset,
        callbacks = [model_checkpoint_callback]
        )

print("Tensorflow version " + tf.__version__)

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver('grpc://10.128.0.3:8470',
        zone = 'us-central1-f',
        project = 'bubbly-delight-378605'  # TPU detection
        )
    print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
except ValueError:
    raise BaseException('ERROR: Not connected to a TPU runtime')

tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

tpu_strategy.run(train_model)


    