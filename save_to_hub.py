from transformers import TFPegasusForConditionalGeneration
from model import get_config
import os
import tensorflow as tf
# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
from sentencepiece_tokenizer import fetch_tokenizer
LOAD_CKPT_PATH = os.getenv("LOAD_CKPT_PATH")
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE"))
repo_name = os.getenv("SAVE_TO_REPO")

if __name__=="__main__":
    model = TFPegasusForConditionalGeneration(get_config(VOCAB_SIZE))
    model.load_weights(LOAD_CKPT_PATH)
    # EVALUATE SOMETHING HERE
    print("Pushing model to hub")
    model.push_to_hub(repo_name)
    tokenizer = fetch_tokenizer()
    print("Pushing tokenizer to hub")
    tokenizer.push_to_hub(repo_name)

