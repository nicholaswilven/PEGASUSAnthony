#from __future__ import absolute_import
import tensorflow as tf
tf.random.set_seed(42)
print("Tensorflow version " + tf.__version__)
import os
import time
from datetime import datetime
import random
random.seed(42)
from transformers import TFPegasusForConditionalGeneration

from utils.parse_records import get_dataset, get_dataset_partitions_tf
from utils.sentencepiece_tokenizer import fetch_tokenizer
from utils.model_config import get_config

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")
REPO_NAME = os.getenv("SAVE_TO_REPO")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE"))

# Parse sys args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--exp_name", help = "experiment name, for checkpoint name", default = datetime.now().strftime("%Y-%m-%d_%H:%M:%S"),
                    type=str)
parser.add_argument("--mode", help = "pretrain or finetune", default = "pretrain", type = str)
parser.add_argument("--num_files", help = "number of files used in training", type = int)
parser.add_argument("--batch_size", help = "batch_size training", default = 128, type = int)
parser.add_argument("--epochs", help = "epochs training", default = 2, type = int)
parser.add_argument("--vocab_size", help = "vocab size model and tokenizer", default = VOCAB_SIZE, type = int)
parser.add_argument("--learning_rate", help = "learning rate training", default = 0.000001, type = float)
parser.add_argument("--load_ckpt_path", help = "path to load model weights", type = str)
parser.add_argument("--repo_name", help = "hugging face repo name to push", default = REPO_NAME, type = str)
parser.add_argument("--learning_rate_decay", help = "scale exp(-lrd) for epoch > 3", default = 0.1, type = float)
args = parser.parse_args()

# Load Dataset
AUTO = tf.data.experimental.AUTOTUNE
tfr_dir = f"gs://{GCS_BUCKET_NAME}"+"/records/{}/{}_{}.tfrecord"
if args.mode == "pretrain":
    # Pretrain dataset is split into 3 folders in buckets
    f1 = [tfr_dir.format("oscar_32k",args.mode,idx) for idx in range(48)]
    f2 = [tfr_dir.format("news_32k",args.mode,idx) for idx in range(144)]
    f3 = [tfr_dir.format("oscar_32k_trunc",args.mode,idx) for idx in range(35)]
    f = f1+f2+f3
    # Shuffle filename list for better randomness
    random.shuffle(f)
    dataset = get_dataset(files = f).prefetch(AUTO)
elif args.mode == "finetune":
    f1 = [tfr_dir.format("indosum_32k",args.mode,idx) for idx in range(4)]
    f2 = [tfr_dir.format("liputan6_32k",args.mode,idx) for idx in range(144)]
    f3 = [tfr_dir.format("oscar_32k_trunc",args.mode,idx) for idx in range(35)]
    f = f1+f2+f3
    # Shuffle filename list for better randomness
    random.shuffle(f)
    dataset = get_dataset(files = f).prefetch(AUTO)

train_dataset, val_dataset = get_dataset_partitions_tf(dataset, BATCH_SIZE = args.batch_size)

# Callbacks for Learning rate decay, Tensorboard, EarlyStopping and Checkpoint
checkpoint_filepath = f"gs://{GCS_BUCKET_NAME}/checkpoints/{args.exp_name}/{args.mode}-"+"weights-{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}"

def scheduler(epoch, lr):
    if epoch < 4:
        return lr
    else:
        return lr * tf.math.exp(-1*args.learning_rate_decay)

model_callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),
    tf.keras.callbacks.TerminateOnNaN(),
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

with tpu_strategy.scope():
    model = TFPegasusForConditionalGeneration(get_config(args.vocab_size))
    
    if args.load_ckpt_path != None:
        model.load_weights(args.load_ckpt_path)

    model.compile(optimizer = tf.keras.optimizers.Adafactor(learning_rate = args.learning_rate),metrics = ["accuracy"])

    print('Start training!')
    model.fit(train_dataset,
        epochs = args.epochs,
        verbose = 1,
        validation_data = val_dataset,
        callbacks = model_callback
        )
    
    if args.push_to_hub:
        model.push_to_hub(args.repo_name+"_"+args.mode)
        tokenizer = fetch_tokenizer()
        tokenizer.push_to_hub(args.repo_name+"_"+args.mode)
