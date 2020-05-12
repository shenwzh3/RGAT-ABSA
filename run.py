# coding=utf-8
import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import random

import numpy as np
import pandas as pd
import torch
from transformers import (BertConfig, BertForTokenClassification,
                                  BertTokenizer)
from torch.utils.data import DataLoader

from datasets import load_datasets_and_vocabs
from model import (Aspect_Text_GAT_ours,
                    Pure_Bert, Aspect_Bert_GAT, Aspect_Text_GAT_only)
from trainer import train

logger = logging.getLogger(__name__)

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--dataset_name', type=str, default='rest',
                        choices=['rest', 'laptop', 'twitter'],
                        help='Choose absa dataset.')
    parser.add_argument('--output_dir', type=str, default='/data1/SHENWZH/ABSA_online/data/output-gcn',
                        help='Directory to store intermedia data, such as vocab, embeddings, tags_vocab.')
    parser.add_argument('--num_classes', type=int, default=3,
                        help='Number of classes of ABSA.')


    parser.add_argument('--cuda_id', type=str, default='3',
                        help='Choose which GPUs to run')
    parser.add_argument('--seed', type=int, default=2019,
                        help='random seed for initialization')

    # Model parameters
    parser.add_argument('--glove_dir', type=str, default='/data1/SHENWZH/wordvec',
                        help='Directory storing glove embeddings')
    parser.add_argument('--bert_model_dir', type=str, default='/data1/SHENWZH/models/bert_base',
                        help='Path to pre-trained Bert model.')
    parser.add_argument('--pure_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')
    parser.add_argument('--gat_bert', action='store_true',
                        help='Cat text and aspect, [cls] to predict.')

    parser.add_argument('--highway', action='store_true',
                        help='Use highway embed.')

    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of layers of bilstm or highway or elmo.')


    parser.add_argument('--add_non_connect',  type= bool, default=True,
                        help='Add a sepcial "non-connect" relation for aspect with no direct connection.')
    parser.add_argument('--multi_hop',  type= bool, default=True,
                        help='Multi hop non connection.')
    parser.add_argument('--max_hop', type = int, default=4,
                        help='max number of hops')


    parser.add_argument('--num_heads', type=int, default=6,
                        help='Number of heads for gat.')
    
    parser.add_argument('--dropout', type=float, default=0,
                        help='Dropout rate for embedding.')


    parser.add_argument('--num_gcn_layers', type=int, default=1,
                        help='Number of GCN layers.')
    parser.add_argument('--gcn_mem_dim', type=int, default=300,
                        help='Dimension of the W in GCN.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2,
                        help='Dropout rate for GCN.')
    # GAT
    parser.add_argument('--gat', action='store_true',
                        help='GAT')
    parser.add_argument('--gat_our', action='store_true',
                        help='GAT_our')
    parser.add_argument('--gat_attention_type', type = str, choices=['linear','dotprod','gcn'], default='dotprod',
                        help='The attention used for gat')

    parser.add_argument('--embedding_type', type=str,default='glove', choices=['glove','bert'])
    parser.add_argument('--embedding_dim', type=int, default=300,
                        help='Dimension of glove embeddings')
    parser.add_argument('--dep_relation_embed_dim', type=int, default=300,
                        help='Dimension for dependency relation embeddings.')

    parser.add_argument('--hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--final_hidden_size', type=int, default=300,
                        help='Hidden size of bilstm, in early stage.')
    parser.add_argument('--num_mlps', type=int, default=2,
                        help='Number of mlps in the last of model.')

    # Training parameters
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=32, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for Adam.")
    
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=30.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps(that update the weights) to perform. Override num_train_epochs.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    
    return parser.parse_args()


def check_args(args):
    '''
    eliminate confilct situations
    
    '''
    logger.info(vars(args))
        


def main():
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    
    # Parse args
    args = parse_args()
    check_args(args)

    # Setup CUDA, GPU training
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    logger.info('Device is %s', args.device)

    # Set seed
    set_seed(args)

    # Bert, load pretrained model and tokenizer, check if neccesary to put bert here
    if args.embedding_type == 'bert':
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_dir)
        args.tokenizer = tokenizer

    # Load datasets and vocabs
    train_dataset, test_dataset, word_vocab, dep_tag_vocab, pos_tag_vocab= load_datasets_and_vocabs(args)

    # Build Model
    # model = Aspect_Text_Multi_Syntax_Encoding(args, dep_tag_vocab['len'], pos_tag_vocab['len'])
    if args.pure_bert:
        model = Pure_Bert(args)
    elif args.gat_bert:
        model = Aspect_Bert_GAT(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # R-GAT + Bert
    elif args.gat_our:
        model = Aspect_Text_GAT_ours(args, dep_tag_vocab['len'], pos_tag_vocab['len']) # R-GAT with reshaped tree
    else:
        model = Aspect_Text_GAT_only(args, dep_tag_vocab['len'], pos_tag_vocab['len'])  # original GAT with reshaped tree

    model.to(args.device)
    # Train
    _, _,  all_eval_results = train(args, train_dataset, model, test_dataset)

    if len(all_eval_results):
        best_eval_result = max(all_eval_results, key=lambda x: x['acc']) 
        for key in sorted(best_eval_result.keys()):
            logger.info("  %s = %s", key, str(best_eval_result[key]))


if __name__ == "__main__":
    main()

