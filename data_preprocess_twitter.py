'''
Biaffine Dependency parser from AllenNLP
'''
import argparse
import json
import os
import re
import sys

from allennlp.predictors.predictor import Predictor
from lxml import etree
from nltk.tokenize import TreebankWordTokenizer
from tqdm import tqdm

MODELS_DIR = '/data1/yangyy/models'
model_path = os.path.join(
    MODELS_DIR, "biaffine-dependency-parser-ptb-2018.08.23.tar.gz")


def parse_args():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument('--model_path', type=str, default=model_path,
                        help='Path to biaffine dependency parser.')
    parser.add_argument('--data_path', type=str, default='/data1/SHENWZH/ABSA_online/data/twitter',
                        help='Directory of where semeval14 or twiiter data held.')
    return parser.parse_args()


sentiment_map = {0: 'neutral', 1: 'positive', -1: 'negative'}


def read_file(file_name):
    '''
    Read twitter data and extract text and store.
    return sentences of [sentence, aspect_sentiment, from_to]
    '''
    with open(file_name, 'r') as f:
        data = f.readlines()
        data = [d.strip('\n') for d in data]
    # list of dict {text, aspect, sentiment}
    sentences = []
    idx = 0
    while idx < len(data):
        text = data[idx]
        idx += 1
        aspect = data[idx]
        idx += 1
        sentiment = data[idx]
        idx += 1
        sentence = get_sentence(text, aspect, sentiment)
        sentences.append(sentence)
    print(file_name, len(sentences))
    with open(file_name.replace('.raw', '.txt'), 'w') as f:
        for sentence in sentences:
            f.write(sentence['sentence'] + '\n')

    return sentences


def get_sentence(text, aspect, sentiment):
    sentence = dict()
    sentence['sentence'] = text.replace('$T$', aspect)
    sentence['aspect_sentiment'] = [[aspect, sentiment_map[int(sentiment)]]]
    frm = text.split().index('$T$')
    to = frm + len(aspect.split())
    sentence['from_to'] = [[frm, to]]
    return sentence


def text2docs(file_path, predictor):
    '''
    Annotate the sentences from extracted txt file using AllenNLP's predictor.
    '''
    with open(file_path, 'r') as f:
        sentences = f.readlines()
    docs = []
    print('Predicting dependency information...')
    for i in tqdm(range(len(sentences))):
        docs.append(predictor.predict(sentence=sentences[i]))

    return docs


def dependencies2format(doc):  # doc.sentences[i]
    '''
    Format annotation: sentence of keys
                                - tokens
                                - tags
                                - predicted_dependencies
                                - predicted_heads
                                - dependencies
    '''
    sentence = {}
    sentence['tokens'] = doc['words']
    sentence['tags'] = doc['pos']
    # sentence['energy'] = doc['energy']
    predicted_dependencies = doc['predicted_dependencies']
    predicted_heads = doc['predicted_heads']
    sentence['predicted_dependencies'] = doc['predicted_dependencies']
    sentence['predicted_heads'] = doc['predicted_heads']
    sentence['dependencies'] = []
    for idx, item in enumerate(predicted_dependencies):
        dep_tag = item
        frm = predicted_heads[idx]
        to = idx + 1
        sentence['dependencies'].append([dep_tag, frm, to])

    return sentence


def get_dependencies(file_path, predictor):
    docs = text2docs(file_path, predictor)
    sentences = [dependencies2format(doc) for doc in docs]
    return sentences

def syntaxInfo2json(sentences, sentences_with_dep, file_name):
    json_data = []
    tk = TreebankWordTokenizer()
    # mismatch_counter = 0
    for idx, sentence in enumerate(sentences):
        sentence['tokens'] = sentences_with_dep[idx]['tokens']
        sentence['tags'] = sentences_with_dep[idx]['tags']
        sentence['predicted_dependencies'] = sentences_with_dep[idx]['predicted_dependencies']
        sentence['dependencies'] = sentences_with_dep[idx]['dependencies']
        sentence['predicted_heads'] = sentences_with_dep[idx]['predicted_heads']
        # sentence['energy'] = sentences_with_dep[idx]['energy']
        json_data.append(sentence)
    
    with open(file_name.replace('.txt', '_biaffine.json'), 'w') as f:
        json.dump(json_data, f)
    print('done', len(json_data))


def main():
    args = parse_args()

    predictor = Predictor.from_path(args.model_path)

    train_file = os.path.join(args.data_path, 'train.raw')
    test_file = os.path.join(args.data_path, 'test.raw')

    # raw -> txt
    train_sentences = read_file(train_file)
    test_sentences = read_file(test_file)

    # Get dependency annotation
    train_sentences_with_dep = get_dependencies(os.path.join(args.data_path, 'train.txt'), predictor)
    test_sentences_with_dep = get_dependencies(os.path.join(args.data_path, 'test.txt'), predictor)
    print(len(train_sentences), len(test_sentences))

    # to json
    syntaxInfo2json(train_sentences, train_sentences_with_dep, os.path.join(args.data_path, 'train.txt'))
    syntaxInfo2json(test_sentences, test_sentences_with_dep, os.path.join(args.data_path, 'test.txt'))

if __name__ == "__main__":
    main()
