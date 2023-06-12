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

def fetch_tokenizer(tokenizer_type : str = TOKENIZER_TYPE,
                    path_to_tokenizer : str = PATH_TO_TOKENIZER):
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
    return tokenizer
    
def _tokenize_inputs(examples : dict,
                     tokenizer,
                     MODEL_MAX_LENGTH : int = MODEL_MAX_LENGTH,
                     MAX_SUMMARY_LENGTH : int = MAX_SUMMARY_LENGTH):
        """Function to tokenize inputs, compatible with hf datasets mapping
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

