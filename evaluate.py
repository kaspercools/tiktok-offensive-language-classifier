import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tiktok_text_processing
import ml_utils
import time
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import sys
import pandas as pd
import sys, getopt
import random

def evaluate_samples(tokenizer, df):
    
    df['full_text'] = df['full_text'].apply(tiktok_text_processing.replace_emoji_w_token)

    df['label'] = df['label'].apply(lambda x: 1 if x =="Offensive" else 0)

    token_id, attention_masks = ml_utils.encode_data(tokenizer, df['full_text'])
    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)   
    labels = torch.tensor(df.label.values.astype(int))

    val_set = TensorDataset(token_id, attention_masks, labels)

    validation_dataloader = DataLoader(
                val_set,
                sampler = SequentialSampler(val_set),
                batch_size = 32
            )

    accuracy, precision, recall, specificity, predictions, labels = ml_utils.evaluate(device, model, validation_dataloader)

    return {
        'F1':'{:.4f}'.format(F1),
        'accuracy':'{:.4f}'.format(accuracy),
        'precision':'{:.4f}'.format(precision),
        'recall':'{:.4f}'.format(recall),
        'specificity':'{:.4f}'.format(specificity),
        'labels':labels,
        'predictions': predictions
    }
def run(model_path:str)-> None:
    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case = True
    )

    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )
    tokenizer = ml_utils.generate_tokenizer(True, True)

    model.resize_token_embeddings(len(tokenizer))

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    device = torch.device('cpu')
    model.eval()

    for i in range(100):
        df = pd.read_csv('public_data_labeled.csv').sample(2000)
        training_result = (evaluate_samples(tokenizer, df))
        ml_utils.print_training_result(training_result)
        timestr = time.strftime("%Y%m%d_%H%M%S")
        filepath = 'eval_result/'+timestr+'_eval'
        ml_utils.save_model(model, [training_result], filepath, True)

if __name__ == '__main__':
    run('data/20230426_002347.bin')
