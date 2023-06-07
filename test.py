import pandas as pd
import numpy as np
import argparse
import warnings
warnings.filterwarnings(action='ignore')

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler # RandomSampler
from keras.preprocessing.sequence import pad_sequences
import transformers
from transformers import BertTokenizerFast, BertForTokenClassification

from utils import tr_te_split, undersampling, massage, tokenize_and_preserve_labels
from evaluate import evaluate


def parse_args():
    parser = argparse.ArgumentParser(description='training BERT')
    parser.add_argument(
        '--api_url', type=str, default=None)
    parser.add_argument(
        '--secret_key', type=str, default=None)
    parser.add_argument(
        '--json_path', type=str, default='data/json_result')
    parser.add_argument(
        '--max_length', type=int, default=256)
    parser.add_argument(
        '--batch_size', type=int, default=4)
    parser.add_argument(
        '--learning_rate', type=float, default=3e-5)
    parser.add_argument(
        '--eps', type = float, default = 1e-8)
    parser.add_argument(
        '--epochs', type=int, default=10)
    parser.add_argument(
        '--max_grad_norm', type=float, default=1.0)
    parser.add_argument(
        '--check_point', type=str, default='NER_best.pth')
    
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('## Load Dataset ##')
    df = pd.read_csv('data/preprocessed.csv', encoding="latin1").fillna(method="ffill")
    print(' ')

    print('## Load Tokenizer ##')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    print(' ')

    _, df_test = tr_te_split(df)

    df_test.reset_index(drop = True, inplace = True)

    labels = [i.split() for i in df_test['labels'].values.tolist()]

    # Value of Label
    unique_labels = set() 
    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]
    unique_labels.add('PAD')

    # Get corpus
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}

    sentences = [df_test.text[i].split(' ')[:-1] for i in range(df_test.shape[0])]
    lb = [i.split() for i in df_test['labels'].values.tolist()]
    labels = [[labels_to_ids[label] for label in sublist] for sublist in lb]

    tokenized_texts_and_labels = [tokenize_and_preserve_labels(tokenizer, sent, labs) for sent, labs in zip(sentences, labels)]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    tag_values = list(unique_labels)

    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=args.max_length, dtype="long", value=0.0,
                            truncating="post", padding="post")

    tags = pad_sequences([[labels_to_ids.get(l) for l in lab] for lab in lb],
                        maxlen=args.max_length, value=labels_to_ids["PAD"], padding="post",
                        dtype="long", truncating="post")
    
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    te_inputs = torch.tensor(input_ids)
    te_tags = torch.tensor(tags)
    te_masks = torch.tensor(attention_masks)

    te_data = TensorDataset(te_inputs, te_masks, te_tags)
    te_sampler = RandomSampler(te_data)
    te_dataloader = DataLoader(te_data, sampler=te_sampler, batch_size=args.batch_size)

    print('## Finish Load Dataset ##')
    print('')

    print('## Load Pretrained BERT ##')
    model = BertForTokenClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(labels_to_ids),
                                                        output_attentions = False,
                                                        output_hidden_states = False)
    PATH = 'pretrained_model/{}'.format(args.check_point)
    model.load_state_dict(torch.load(PATH), strict=False)
    print('')

    model.to(device)

    print('## Test!!! ##')
    evaluate(model, te_dataloader, tag_values)


if __name__ == '__main__':
    main()