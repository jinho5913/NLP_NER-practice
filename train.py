import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings(action='ignore')

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from seqeval.metrics import f1_score, accuracy_score
import transformers
from transformers import BertTokenizerFast, BertForTokenClassification, AdamW, logging, get_linear_schedule_with_warmup
logging.set_verbosity_error()

from utils import tokenize_and_preserve_labels, push_lst


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
        '--batch_size', type=int, default=16)
    parser.add_argument(
        '--learning_rate', type=float, default=3e-5)
    parser.add_argument(
        '--eps', type = float, default = 1e-8)
    parser.add_argument(
        '--epochs', type=int, default=10)
    parser.add_argument(
        '--max_grad_norm', type=float, default=1.0)
    
    args = parser.parse_args()

    return args


def train_model(model, train, val, optimizer, scheduler, tag_values, epochs, max_grad_norm):
    loss_values, validation_loss_values = [], []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    best_acc = 0
    tr_loss, val_loss = [], []
    tr_acc, val_acc = [], []
    tr_f1, val_f1 = [], []
    for epoch_num in tqdm(range(epochs)):
        # ========================================
        #               Training
        # ========================================

        model.train()
        total_loss = 0

        # Training loop
        for step, batch in enumerate(train):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch # 배치 별 input_id, mask, label
            
            model.zero_grad()
            
            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels) # output of model
            loss = outputs[0]
            loss.backward()
            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=max_grad_norm)
            optimizer.step()
            scheduler.step() # 설정된 스케쥴러 사용

        avg_train_loss = total_loss / len(train)
        loss_values.append(avg_train_loss)
        
        train_predictions, train_labels = [], []
        for batch in train:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)
            
            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            train_predictions.extend([list(p) for p in np.argmax(logits, axis=2)]) # softmax로 나온 값들 중 argmax를 통해 prediction 값 재할당
            train_labels.extend(label_ids)

        train_pred_tags = [tag_values[p_i] for p, l in zip(train_predictions, train_labels) for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"] # PAD token이 아닌 값들 이중 리스트화로 합침 -> 평가지표에 사용
        train_true_tags = [tag_values[l_i] for l in train_labels for l_i in l if tag_values[l_i] != "PAD"] # PAD token이 아닌 값들 이중 리스트화로 합침 -> 평가지표에 사용

        # ========================================
        #               Validation
        # ========================================

        model.eval()
        eval_loss = 0
        val_predictions , true_labels = [], []
        for batch in val:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():

                outputs = model(b_input_ids, token_type_ids=None,
                                attention_mask=b_input_mask, labels=b_labels)

            logits = outputs[1].detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            eval_loss += outputs[0].mean().item()
            val_predictions.extend([list(p) for p in np.argmax(logits, axis=2)])
            true_labels.extend(label_ids)

        eval_loss = eval_loss / len(val)
        validation_loss_values.append(eval_loss)
        val_pred_tags = [tag_values[p_i] for p, l in zip(val_predictions, true_labels) for p_i, l_i in zip(p, l) if tag_values[l_i] != "PAD"]
        val_valid_tags = [tag_values[l_i] for l in true_labels for l_i in l if tag_values[l_i] != "PAD"]

        tr_loss.append(avg_train_loss)
        val_loss.append(eval_loss)
        tr_acc.append(accuracy_score(train_pred_tags, train_true_tags))
        val_acc.append(accuracy_score(val_pred_tags, val_valid_tags))
        tr_f1.append(f1_score([train_pred_tags], [train_true_tags]))
        val_f1.append(f1_score([val_pred_tags], [val_valid_tags]))

        
        print(f'Epoch: {epoch_num + 1} | Train_Loss: {avg_train_loss} | Train_Accuracy: {accuracy_score(train_pred_tags, train_true_tags)} | Train_F1 score: {f1_score([train_pred_tags], [train_true_tags])} | Val_Loss: {eval_loss} | Val_Accuracy: {accuracy_score(val_pred_tags, val_valid_tags)} | Val_F1 score: {f1_score([val_pred_tags], [val_valid_tags])}')

        print('Save Model...')
        if accuracy_score(val_pred_tags, val_valid_tags) > best_acc: # validation accuracay가 이전보다 높을 경우 model save
            best_acc = accuracy_score(val_pred_tags, val_valid_tags)
            PATH = 'pretrained_model/NER_best.pth'
            torch.save(model.state_dict(), PATH)
        else:
            pass

    tr_loss, val_loss, tr_acc, val_acc, tr_f1, val_f1 = np.array(tr_loss), np.array(val_loss), np.array(tr_acc), np.array(val_acc), np.array(tr_f1), np.array(val_f1)

    push_lst(tr_loss)
    push_lst(val_loss)
    push_lst(tr_acc)
    push_lst(val_acc)
    push_lst(tr_f1)
    push_lst(val_f1)



def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('## Load Dataset ##')
    df_train = pd.read_csv('data/preprocessed.csv', encoding="latin1").fillna(method="ffill")
    print(' ')

    print('## Load Tokenizer ##')
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    print(' ')

    labels = [i.split() for i in df_train['labels'].values.tolist()]

    # Value of Label
    unique_labels = set() 
    for lb in labels:
        [unique_labels.add(i) for i in lb if i not in unique_labels]
    unique_labels.add('PAD')

    # Get corpus
    labels_to_ids = {k: v for v, k in enumerate(unique_labels)}

    sentences = [df_train.text[i].split(' ')[:-1] for i in range(df_train.shape[0])] # sentence 리스트
    lb = [i.split() for i in df_train['labels'].values.tolist()]
    labels = [[labels_to_ids[label] for label in sublist] for sublist in lb] # label 리스트

    tokenized_texts_and_labels = [tokenize_and_preserve_labels(tokenizer, sent, labs) for sent, labs in zip(sentences, labels)]

    tokenized_texts = [token_label_pair[0] for token_label_pair in tokenized_texts_and_labels]
    labels = [token_label_pair[1] for token_label_pair in tokenized_texts_and_labels]

    tag_values = list(unique_labels) # unique of label

    # max_length 까지 padding
    input_ids = pad_sequences([tokenizer.convert_tokens_to_ids(txt) for txt in tokenized_texts],
                            maxlen=args.max_length, dtype="long", value=0.0,
                            truncating="post", padding="post")

    tags = pad_sequences([[labels_to_ids.get(l) for l in lab] for lab in lb],
                        maxlen=args.max_length, value=labels_to_ids["PAD"], padding="post",
                        dtype="long", truncating="post")
    
    # attention mask 설정
    attention_masks = [[float(i != 0.0) for i in ii] for ii in input_ids]

    tr_inputs, val_inputs, tr_tags, val_tags = train_test_split(input_ids, tags, random_state=2023, test_size=0.1)
    tr_masks, val_masks, _, _ = train_test_split(attention_masks, input_ids, random_state=2023, test_size=0.1)

    tr_inputs = torch.tensor(tr_inputs)
    val_inputs = torch.tensor(val_inputs)
    tr_tags = torch.tensor(tr_tags)
    val_tags = torch.tensor(val_tags)
    tr_masks = torch.tensor(tr_masks)
    val_masks = torch.tensor(val_masks)

    # Set Dataloader
    train_data = TensorDataset(tr_inputs, tr_masks, tr_tags)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    val_data = TensorDataset(val_inputs, val_masks, val_tags)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=args.batch_size)
    print('## Finish Load Dataset ##')
    print('')

    # Initialize Model
    model = BertForTokenClassification.from_pretrained("bert-base-uncased",
                                                        num_labels=len(labels_to_ids),
                                                        output_attentions = False,
                                                        output_hidden_states = False)
    print('## Pretrained BERT ##')
    print('')

    model.to(device)

    FULL_FINETUNING = True
    if FULL_FINETUNING:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay_rate': 0.0}
        ]
    else:
        param_optimizer = list(model.classifier.named_parameters())
        optimizer_grouped_parameters = [{"params": [p for n, p in param_optimizer]}]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.eps)

    epochs = 10
    max_grad_norm = 1.0

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, tag_values, args.epochs, args.max_grad_norm)


if __name__ == '__main__':
    main()