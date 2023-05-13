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
parser.add_argument("--vocab_size", help = "vocab size model and tokenizer", default = 32103, type=int)
parser.add_argument("--learning_rate", help = "learning rate training", default = 0.0005, type=float)
parser.add_argument("--load_ckpt_path", help = "path toload model weights", type=str)
parser.add_argument("--from_pretrain", help = "alternative for training mode", default=False, type=bool)
parser.add_argument("--learning_rate_decay", help = "scale exp(-lrd) for epoch > 2", default=0.15, type=float)
args = parser.parse_args()

# Load Dataset
if args.num_files == None:
    if args.mode == "pretrain":
        num_files = PRETRAIN_NUM_FILES
    elif args.mode == "finetune":
        num_files = FINETUNE_NUM_FILES
else:
    num_files = args.num_files

# Mixed Datasets
tfr_dir = f"gs://{GCS_BUCKET_NAME}"+"/records/{}/pretrain_{}.tfrecord"
f1 = [tfr_dir.format("oscar_32k",idx) for idx in range(48)]
f2 = [tfr_dir.format("news_32k",idx) for idx in range(144)]
f3 = [tfr_dir.format("oscar_32k_trunc",idx) for idx in range(35)]
f = f1+f2+f3
import random
random.shuffle(f)
AUTO = tf.data.experimental.AUTOTUNE
dataset = get_dataset(files = f).prefetch(AUTO)

train_dataset, val_dataset = get_dataset_partitions_tf(dataset, args.batch_size,val_size=32000)

# Callbacks for Learning rate decay, Tensorboard, EarlyStopping and Checkpoint
checkpoint_filepath = f"gs://{GCS_BUCKET_NAME}/checkpoints/{args.exp_name}/{args.mode}-"+"weights-{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}"

def scheduler(epoch, lr):
    if epoch < 2:
        return lr
    else:
        return lr * tf.math.exp(-1*args.learning_rate_decay)

model_callback = [tf.keras.callbacks.LearningRateScheduler(scheduler),
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
    if args.from_pretrain:
        model = TFPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
        # Train embedding layer only
        model.layers[0].encoder.trainable = False
        model.layers[0].decoder.trainable = False
        model.layers[0].shared.trainable =  True

        # Reinitialize embeddings
        x = model.layers[0].shared.embeddings_initializer([96103,1024])
        model.layers[0].shared.set_weights([x])
    else:
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
    
    if args.from_pretrain:
        # save embedding layer only
        np.save(exp_name+"_embedding_layer",model.layers[0].shared.get_weights()[0])