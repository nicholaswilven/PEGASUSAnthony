#from __future__ import absolute_import
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
import os
import time
from datetime import datetime



from parse_records import get_dataset, get_dataset_partitions_tf
from transformers import TFPegasusForConditionalGeneration, PegasusConfig

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()

GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Parse sys args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", help = "experiment name, for checkpoint name", default = datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                    type=str)
parser.add_argument("--mode", help = "pretrain or finetune", default = "pretrain",
                    type=str)
parser.add_argument("--num_files", help = "number of files used in training", default = 276,
                    type=int)
parser.add_argument("--batch_size", help = "batch_size training", default = 128,
                    type=int)
parser.add_argument("--epochs", help = "epochs training", default = 32,
                    type=int)
parser.add_argument("--vocab_size", help = "vocab size model and tokenizer", default = 64103,
                    type=int)
parser.add_argument("--learning_rate", help = "learning rate training", default = 0.1,
                    type=float)
args = parser.parse_args()

# Preparation for training
AUTO = tf.data.experimental.AUTOTUNE
dataset = get_dataset(mode  = args.mode, num_files = args.num_files).prefetch(AUTO)
train_dataset, val_dataset = get_dataset_partitions_tf(dataset, args.batch_size)

checkpoint_filepath = (f"gs://{GCS_BUCKET_NAME}/checkpoints/{args.exp_name}/{args.mode}-weights-"+"e{epoch}-l{val_loss}.ckpt")

model_callback = [
    tf.keras.callbacks.TensorBoard(log_dir='./logs/'+args.exp_name),
    tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    min_delta = 0,
                    patience = 2,
                    verbose = 1,
                    mode="auto",
                    baseline=None,
                    restore_best_weights=False,
                    start_from_epoch = 10),
    tf.keras.callbacks.ModelCheckpoint(
                    filepath = checkpoint_filepath,
                    save_weights_only = True,
                    monitor = "val_loss",
                    mode = 'auto',
                    verbose = 1,
                    save_best_only = True,
                    initial_value_threshold = 4
                    )]

configuration = PegasusConfig()
configuration.vocab_size = args.vocab_size

tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

tf.profiler.experimental.server.start(6000)

with tpu_strategy.scope():
    model = TFPegasusForConditionalGeneration(configuration)
    model.compile(optimizer = tf.keras.optimizers.experimental.Adafactor(learning_rate = args.learning_rate), metrics = ["accuracy"])

    print('Start training!')
    model.fit(train_dataset,
        epochs = args.epochs,
        verbose = 1,
        validation_data = val_dataset,
        callbacks = model_callback
        )


    