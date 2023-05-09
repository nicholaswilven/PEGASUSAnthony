#from __future__ import absolute_import
import tensorflow as tf
print("Tensorflow version " + tf.__version__)
import os
import time
from datetime import datetime

from parse_records import get_dataset, get_dataset_partitions_tf
from transformers import TFPegasusForConditionalGeneration
from model import get_config

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
PRETRAIN_NUM_FILES = int(os.getenv("PRETRAIN_NUM_FILES"))
FINETUNE_NUM_FILES = int(os.getenv("FINETUNE_NUM_FILES"))
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

# Parse sys args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", help = "experiment name, for checkpoint name", default = datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                    type=str)
parser.add_argument("--mode", help = "pretrain or finetune", default = "pretrain", type=str)
parser.add_argument("--num_files", help = "number of files used in training", type=int)
parser.add_argument("--batch_size", help = "batch_size training", default = 128, type=int)
parser.add_argument("--epochs", help = "epochs training", default = 40, type=int)
parser.add_argument("--vocab_size", help = "vocab size model and tokenizer", default = 16103, type=int)
parser.add_argument("--learning_rate", help = "learning rate training", default = 0.0005, type=float)
parser.add_argument("--load_ckpt_path", help = "path toload model weights", type=str)
args = parser.parse_args()

# Load Dataset
if args.num_files == None:
    if args.mode == "pretrain":
        num_files = PRETRAIN_NUM_FILES
    elif args.mode == "finetune":
        num_files = FINETUNE_NUM_FILES
else:
    num_files = args.num_files

AUTO = tf.data.experimental.AUTOTUNE
dataset = get_dataset(mode  = args.mode, num_files = num_files).prefetch(AUTO)
train_dataset, val_dataset = get_dataset_partitions_tf(dataset, args.batch_size,val_size=32000)

# Callbacks for Tensorboard, EarlyStopping and Checkpoint
checkpoint_filepath = f"gs://{GCS_BUCKET_NAME}/checkpoints/{args.exp_name}/{args.mode}-"+"weights-{epoch:02d}-{val_loss:.3f}"

model_callback = [
    tf.keras.callbacks.TensorBoard(log_dir='./tensorboard_logs/'+args.exp_name),
    tf.keras.callbacks.EarlyStopping(
                    monitor = "val_accuracy",
                    min_delta = 0,
                    patience = 2,
                    verbose = 1,
                    mode = "auto",
                    baseline = None,
                    restore_best_weights = False,
                    start_from_epoch = 16),
    tf.keras.callbacks.ModelCheckpoint(
                    filepath = checkpoint_filepath,
                    save_weights_only = True,
                    monitor = "val_accuracy",
                    mode = 'auto',
                    verbose = 1,
                    save_best_only = True,
                    initial_value_threshold = 0.1
                    )]

# Use TPU!
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)
tpu_strategy = tf.distribute.TPUStrategy(tpu)

# This one is for tensorboard
tf.profiler.experimental.server.start(6000)

def scce_with_ls(y, y_hat):
    y = tf.one_hot(tf.cast(y, tf.int32), args.vocab_size)
    return tf.keras.losses.categorical_crossentropy(y, tf.add(y_hat,1e-10), label_smoothing = 0.1)

with tpu_strategy.scope():
    model = TFPegasusForConditionalGeneration(get_config(args.vocab_size))
    if args.load_ckpt_path != None:
        model.load_weights(args.load_ckpt_path)

    # Check use label smooting loss or not
    if args.label_smoothing:
        model.compile(loss = scce_with_ls,
            optimizer = tf.keras.optimizers.Adafactor(learning_rate = args.learning_rate),
            metrics = ["accuracy"])
    else:
        model.compile(optimizer = tf.keras.optimizers.Adafactor(learning_rate = args.learning_rate),
            metrics = ["accuracy"])

    print('Start training!')
    model.fit(train_dataset,
        epochs = args.epochs,
        verbose = 1,
        validation_data = val_dataset,
        callbacks = model_callback
        )