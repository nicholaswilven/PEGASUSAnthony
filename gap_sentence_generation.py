from rouge_score import rouge_scorer
import numpy as np
import random
import re
import nltk
import random
import os
from dotenv import load_dotenv
load_dotenv()
GSG_RATE = float(os.getenv("GSG_RATE"))
RETURN_MASK_RATE = float(os.getenv("RETURN_MASK_RATE"))

def _e_gap_sentence_generation(examples, GSG_RATE = GSG_RATE, RETURN_MASK_RATE = RETURN_MASK_RATE):
    """Function to prepare features for E_GSG task
    Arguments:
    list of text: ["Pegasus is mythical . It is pure white . it names the model ."]
    GGS_RATE: Percentage of sentence masked in input using top ROUGE F1 score
    RETURN_MASK_RATE: Choosen sentence are not masked in input, set 0 for regular GSG
    Returns:
    result :  A dictionary with 2 keys
    input : ["Pegasus is mythical . <mask_1> it names the model ."]
    labels :  ["It is pure white . </s>"] for <mask_1>
    """

    result = {}
    result["input"] = []
    result["labels"] = []

    # Split the documents from the dataset into it's individual sentences
    examples_sentences = [nltk.tokenize.sent_tokenize(doc) for doc in examples['text']]

    scorer = rouge_scorer.RougeScorer(['rouge1']) # Metric used : ROUGE1 F1 score

    for document in examples_sentences: 
        m = len(document)

        num_of_masked_sent = max(1,int(m*GSG_RATE))
        num_of_unmasked_sent = num_of_masked_sent - max(0,int(num_of_masked_sent*RETURN_MASK_RATE))

        score = [0 for i in range(m)]
        predictions = [re.sub(r'[^a-zA-Z0-9]', ' ', document[i]) for i in range(m)] # remove punctuation for better rouge score
          
        for i in range(m):
            references = " ".join([predictions[j] for j in range(m) if j!=i]) # remove ith sentence for ith entry
            score[i] = scorer.score(predictions[i], references)['rouge1'].fmeasure
            
        ind = sorted(np.argpartition(score, -num_of_masked_sent)[-num_of_masked_sent:])
        ind_unmask = random.choices(ind,k = num_of_unmasked_sent)
            
        # build outputs from decided sentences to mask
        input_string = " ".join([document[j] if j not in ind_unmask else "<mask_1>" for j in range(m)])
        labels_string = " ".join([document[j] for j in ind])

        result["input"].append(input_string)
        result["labels"].append(labels_string)
      
    return result