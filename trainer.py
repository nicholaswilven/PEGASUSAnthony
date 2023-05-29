#from __future__ import absolute_import
import tensorflow as tf
tf.random.set_seed(42)
print("Tensorflow version " + tf.__version__)
import os
import time
from datetime import datetime
import random
random.seed(42)

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
parser.add_argument("--epochs", help = "epochs training", default = 50, type=int)
parser.add_argument("--vocab_size", help = "vocab size model and tokenizer", default = 32103, type=int)
parser.add_argument("--learning_rate", help = "learning rate training", default = 0.000001, type=float)
parser.add_argument("--load_ckpt_path", help = "path toload model weights", type=str)
parser.add_argument("--from_pretrain", help = "alternative for training mode", default=False, type=bool)
parser.add_argument("--learning_rate_decay", help = "scale exp(-lrd) for epoch > 2", default=0.1, type=float)
args = parser.parse_args()

# Load Dataset
if args.mode == "pretrain":
    # Pretrain dataset is split into 3 folders in buckets
    tfr_dir = f"gs://{GCS_BUCKET_NAME}"+"/records/{}/pretrain_{}.tfrecord"
    f1 = [tfr_dir.format("oscar_32k",idx) for idx in range(48)]
    f2 = [tfr_dir.format("news_32k",idx) for idx in range(144)]
    f3 = [tfr_dir.format("oscar_32k_trunc",idx) for idx in range(35)]
    f = f1+f2+f3
    # Shuffle filename list for better randomness
    random.shuffle(f)
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = get_dataset(files = f).prefetch(AUTO)
elif args.mode == "finetune":
    num_files = FINETUNE_NUM_FILES
    AUTO = tf.data.experimental.AUTOTUNE
    dataset = get_dataset(mode="finetune",num_files=num_files).prefetch(AUTO)

train_dataset, val_dataset = get_dataset_partitions_tf(dataset, BATCH_SIZE = args.batch_size)

# Callbacks for Learning rate decay, Tensorboard, EarlyStopping and Checkpoint
checkpoint_filepath = f"gs://{GCS_BUCKET_NAME}/checkpoints/{args.exp_name}/{args.mode}-"+"weights-{epoch:02d}-{val_loss:.3f}-{val_accuracy:.3f}"

def scheduler(epoch, lr):
    if epoch < 4:
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
        # Retrain embeddings layer only from pegasus-large, vocab_size follows the args
        model = TFPegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
        config = model.config
        config.vocab_size = args.vocab_size
        encoder_w = model.layers[0].encoder.get_weights()
        decoder_w = model.layers[0].decoder.get_weights()

        model = TFPegasusForConditionalGeneration(config)
        model.build(input_shape = {"input_ids":[128, 512],"decoder_input_ids":[128,256]})
        embed_w = model.layers[0].shared.get_weights()
        encoder_w[0] = embed_w[0]
        decoder_w[0] = embed_w[0]
        model.layers[0].encoder.set_weights(encoder_w)
        model.layers[0].decoder.set_weights(decoder_w)
        model.layers[0].encoder.trainable = False
        model.layers[0].decoder.trainable = False
        model.layers[0].shared.trainable =  True
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
    
    if args.push_to_hub:
        model.push_to_hub("thonyyy")