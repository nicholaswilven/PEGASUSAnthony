# PEGASUSAnthony 
Thesis Project (TBA)

# Infrastructure setup
1. Create TPU VM on GCP (Tensorflow version 2.12.0)
2. Create GCS buckets on GCP

# First time setup
1. python -m venv venv
2. pip install -r requirements.txt
3. python setup.py
4. python tpu-test.py

# Prepare training dataset
1. Dump all text into one .txt file
2. Train sentencepiece tokenizer using train_tokenizer.py
3. Convert all training data to TFRecords using convert_to_records.py

# Training model
1. Run trainer.py

# Bonus: Running process in background (linux)
0. Run task command on script.sh file
1. bash chmod +x script.sh
2. bash ./script.sh &

# Deploy mini showcase using FastAPI
1. Specify model checkpoint path and vocab_size on .env (and model hyperparams on model.py)
2. bash uvicorn app:app
 