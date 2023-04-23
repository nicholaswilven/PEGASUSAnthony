from transformers import PegasusTokenizerFast, PegasusTokenizer
import sentencepiece as spm
import os

# Load Environment Variables from .env
from dotenv import load_dotenv
load_dotenv()
MODEL_MAX_LENGTH = int(os.getenv("MODEL_MAX_LENGTH"))
MAX_SUMMARY_LENGTH = int(os.getenv("MAX_SUMMARY_LENGTH"))
PATH_TO_TOKENIZER = os.getenv("PATH_TO_TOKENIZER")
TOKENIZER_TYPE = os.getenv("TOKENIZER_TYPE")

def fetch_tokenizer(tokenizer_type = TOKENIZER_TYPE, path_to_tokenizer = PATH_TO_TOKENIZER):
    """ Generate tokenizer based on type and default path on .env
    Args: 
        tokenizer_type = 'BPE' for default tokenizeror 'Unigram' for fast tokenizer
    Outputs: PegasusTokenizer object
    """
    if tokenizer_type == "Unigram":
        tokenizer = PegasusTokenizerFast(
                vocab_file = f"{path_to_tokenizer}.model",
                model_max_length = MODEL_MAX_LENGTH
                )
    else:
        tokenizer = PegasusTokenizer(
                vocab_file = f"{path_to_tokenizer}.model",
                model_max_length = MODEL_MAX_LENGTH
                )
    special_tokens_dict = {'additional_special_tokens': ["<s>", "<pad>", "</s>", "<unk>", "<cls>", "<sep>", "<mask_1>","<mask_2>"]}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer
    
def _tokenize_inputs(examples, tokenizer, MODEL_MAX_LENGTH = MODEL_MAX_LENGTH, MAX_SUMMARY_LENGTH = MAX_SUMMARY_LENGTH):
        """Function to tokenize inputs
        Arguments:
        tokenizer: PegasusTokenizer
          examples : A dictionary with 2 keys
            input : ["Pegasus is mythical . <mask_1> it names the model ."]
            labels  :  ["It is pure white . </s>"] for <mask_1>
            MODEL_MAX_LENGTH : Maximum tokens in input
            MAX_SUMMARY_LENGTH : Maximum tokens in output
        Returns:
            examples:  A dictionary with 4 keys 
              input_ids : tokenized masked input ["Pegasus is mythical . <mask_1> it names the model ."]
              labels : tokenized labels ["It is pure white . </s>"]
              attention_mask : attention mask for input_ids (to avoid attention on mask and padding)
              decoder_input_ids : tokenized decoder input (with bos token <pad>) constructed from labels text. ex: ["<pad> It is pure white ."]
        """

        input_string = examples['input']
        labels_string = examples['labels']

        tokenized_inputs = tokenizer(
            input_string,
            return_tensors = "tf",
            padding = 'max_length',
            max_length = MODEL_MAX_LENGTH,
            truncation = True
            )
        
        input_ids = tokenized_inputs.input_ids
        attention_mask = tokenized_inputs.attention_mask

        decoder_input_ids = tokenizer(
            ["<pad> " + single_labels for single_labels in labels_string],
            add_special_tokens = False,
            return_tensors="tf",
            padding = 'max_length',
            max_length = MAX_SUMMARY_LENGTH,
            truncation = True
            ).input_ids

        tokenized_labels = tokenizer(
            labels_string,
            return_tensors = "tf",
            padding = 'max_length',
            max_length = MAX_SUMMARY_LENGTH,
            truncation = True
            ).input_ids
        
        return {'input_ids' : input_ids,
                'attention_mask' : attention_mask,
                'decoder_input_ids' : decoder_input_ids,
                'labels' : tokenized_labels}

if __name__=="__main__":
    # Run to train the tokenizer

    input_file = 'training_tokenizer.txt',
    model_prefix = "PegasusAnthony_64k", 
    vocab_size = 64000,
    model_type = "unigram"
    input_sentence_size = 800000
    spm.SentencePieceTrainer.Train(
        input = input_file,
        model_prefix = model_prefix,
        vocab_size = vocab_size,
        model_type = model_type,
        input_sentence_size = input_sentence_size,
        shuffle_input_sentence = True,
        train_extremely_large_corpus = True
        )