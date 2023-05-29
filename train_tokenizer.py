import sentencepiece as spm

# Parse sys args
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--input_file", help = "txt file for the corpus to train tokenizer", default = "training_tokenizer.txt", type=str)
parser.add_argument("--model_prefix", help = "resulting filename", default = "PegasusAnthony_64k", type=str)
parser.add_argument("--vocab_size", help = "vocab size for tokenizer", default = 16000, type=int)
parser.add_argument("--model_type", help = "unigram or bpe", default = "unigram", type=str)
parser.add_argument("--input_sentence_size", help = "limits the number of sentence size used to train the tokenizer", default = 800000, type=int)
parser.add_argument("--shuffle_input_sentence", help = "shuffle the sentence when training", default= True, type=bool)
parser.add_argument("--train_extremely_large_corpus", help = "True if your corpus is large", default= True, type=bool)
args = parser.parse_args()

if __name__=="__main__":
    # Run to train the tokenizer
    spm.SentencePieceTrainer.Train(
        input = args.input_file,
        model_prefix = args.model_prefix,
        vocab_size = args.vocab_size,
        model_type = args.model_type,
        input_sentence_size = input_sentence_size,
        shuffle_input_sentence = args.shuffle_input_sentence,
        train_extremely_large_corpus = args.train_extremely_large_corpus
        )