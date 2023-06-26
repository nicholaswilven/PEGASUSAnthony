import os
from .convert_to_records import convert_ds_to_records
from .cleaning import text_cleaning
from datasets import load_dataset,concatenate_datasets

def pb(examples):
    examples['input'] = [" ".join(text_cleaning(x).split(" ")[:500]) for x in examples['text']]
    examples['labels'] = [" ".join(text_cleaning(x).split(" ")[:500]) for x in examples['summary']]
    return examples 
    
if __name__=="__main__":
    dataset = concatenate_datasets([load_dataset("csebuetnlp/xlsum",
                           "indonesian",
                           split = "test")])                                    
    dataset =  dataset.map(pb,batched = True,remove_columns = ["id","url","title","summary","title","text"],num_proc = 96)
    convert_ds_to_records(dataset, "xlsum_32k_test",num_proc=96)