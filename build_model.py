import tensorflow as tf
from transformers import TFPegasusForConditionalGeneration, PegasusConfig
import os

from sentencepiece_tokenizer import fetch_tokenizer

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))

tokenizer = fetch_tokenizer()

def build_model(tokenizer = tokenizer):
    # Build model with parameters same as PEGASUS large except vocabulary
    configuration = PegasusConfig()
    configuration.vocab_size = tokenizer.vocab_size

    model = TFPegasusForConditionalGeneration(configuration)
    return model

