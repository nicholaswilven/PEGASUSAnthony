# PEGASUSAnthony 
Thesis Project
A paper named [“PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization”]([url](https://arxiv.org/abs/1912.08777)) appeared in International Conference on Machine Learning 2020 surprised the AI community by the breakthroughs in the abstractive summarization task. 
PEGASUS is a deep learning model that uses a transformer encoder-decoder architecture, which takes in a sequence of input tokens and generates a sequence of output tokens. This is particularly useful for summarization, where the goal is to take a long piece of text and generate a shorter summary.
Made by the Google AI Research team, the model achieved performance close to the state of the art with remarkably less labeled training data, around 1000 data, compared to previous state of the art models that use tens of thousands of labeled training data.
![image](https://github.com/nicholaswilven/PEGASUSAnthony/assets/67919355/62f0ae05-4ac5-4db0-8142-196187b1ed02)



# Infrastructure setup
1. Create TPU VM on GCP (Tensorflow version 2.12.0)
2. Create GCS buckets on GCP

# First time setup
1. python -m venv venv
2. pip install -r requirements.txt
3. python setup.py
4. python tpu-test.py

# Prepare training dataset
0. Preprocess on notebook
1. Dump all text into one .txt file
2. Train sentencepiece tokenizer using train_tokenizer.py
3. Convert all training data to TFRecords using convert_to_records.py

# Training model
1. Run trainer.py

# Running process in background (linux)
0. Run task command on script.sh file
1. bash chmod +x script.sh
2. bash ./script.sh &

# Deploy mini showcase using FastAPI
1. Specify model checkpoint path and vocab_size on .env (and model hyperparams on model.py)
2. bash uvicorn app:app
 
