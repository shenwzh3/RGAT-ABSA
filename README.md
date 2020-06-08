# Relational-Graph-Attention-Network-for-Aspect-based-Sentiment-Analysis
This repo contains the PyTorch implementaion for the paper [Relational Graph Attention Network for Aspect-based Sentiment Analysis](https://arxiv.org/abs/2004.12362).

For any questions about the implementation, plaese email shenwzh3@mail2.sysu.edu.cn.

## Requirements
* Python 3.6.8
* PyTorch 1.2.0
* Transformers https://github.com/huggingface/transformers
* CUDA 9.0

## Perparation
### For Glove Embedding
First, download and unzip GloVe vectors(`glove.840B.300d.zip`) from https://nlp.stanford.edu/projects/glove/. Then change the value of parameter `--glove_dir` to the directory of the word vector file.

### For BERT Embedding
Download the pytorch version pre-trained `bert-base-uncased` model and vocabulary from the link provided by huggingface. Then change the value of parameter `--bert_model_dir` to the directory of the bert model.

## Preprocess
The preprocess codes are in `data_preprocess_semeval.py` and `data_preprocess_twitter.py`. However we already provided the preprocessed datasets with dependency parcing results in `./data/`, so you can skip preprocess.

## Training
Run:
`./run.sh`


