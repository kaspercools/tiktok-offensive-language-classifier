import json
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
import re
import pandas as pd
import numpy as np

from tqdm import trange
import torch, gc

def b_tp(preds, labels):
  '''Returns True Positives (TP): count of correct predictions of actual class 1'''
  return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_fp(preds, labels):
  '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
  return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])

def b_tn(preds, labels):
  '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
  return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_fn(preds, labels):
  '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
  return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])

def b_metrics(preds, labels):
  '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
  preds = np.argmax(preds, axis = 1).flatten()
  labels = labels.flatten()
  tp = b_tp(preds, labels)
  tn = b_tn(preds, labels)
  fp = b_fp(preds, labels)
  fn = b_fn(preds, labels)
  b_accuracy = (tp + tn) / len(labels)
  b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
  b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
  b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
  return b_accuracy, b_precision, b_recall, b_specificity

def clear_gpu_cache():
    # clear cache on re-run   
    gc.collect()
    torch.cuda.empty_cache()

def encode_data(tokenizer, items):
    token_id = []
    attention_masks = []
    item_lenghts = [len(sample) for sample in items]

    item_max_len = 521#np.max(item_lenghts)
    for sample in items:
        encoding_dict = preprocessing(sample, tokenizer, item_max_len)
        token_id.append(encoding_dict['input_ids']) 
        attention_masks.append(encoding_dict['attention_mask'])    

    return (token_id, attention_masks)


def preprocessing(input_text, tokenizer, items_max_len):
  '''
  Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
    - input_ids: list of token ids
    - token_type_ids: list of token type ids
    - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
  '''
  return tokenizer.encode_plus(
                        input_text,
                        add_special_tokens = True,
                        # max_length = 512items_max_len,
                        pad_to_max_length = True,
                        padding='max_length',
                        return_attention_mask = True,
                        truncation = True,
                        return_tensors = 'pt'
                   )

def create_datasets(token_id, attention_masks, labels, val_ratio, batch_size=16):
    # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf

    # Indices of the train and validation splits stratified by labels
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size = val_ratio,
        shuffle = True,
        stratify = labels)

    # Train and validation sets
    train_set = TensorDataset(token_id[train_idx], 
                            attention_masks[train_idx], 
                            labels[train_idx])

    val_set = TensorDataset(token_id[val_idx], 
                            attention_masks[val_idx], 
                            labels[val_idx])


    # Prepare DataLoader
    train_dataloader = DataLoader(
                train_set,
                sampler = RandomSampler(train_set),
                batch_size = batch_size
            )

    validation_dataloader = DataLoader(
                val_set,
                sampler = SequentialSampler(val_set),
                batch_size = batch_size
            )
    
    return (train_dataloader, validation_dataloader)


def read_data_frame(file_path):
    df = pd.read_csv(file_path)
    df.head()
    return df

def remove_links(comment):
    comment = re.sub('http[^\s]+','',comment)
    return comment

def remove_usernames(comment):
    comment = re.sub('@[^\s]+','',comment)    
    return comment
    

def save_model(model, training_results, filepath):
    #todo save results as well
    torch.save(model.state_dict(),filepath)

    # Serializing json
    json_object = json.dumps(training_results, indent=4)
    
    # Writing to sample.json
    with open(filepath+'_results.json', "w") as outfile:
        outfile.write(json_object)

def train(device, model, train_dataloader, validation_dataloader, learning_rate=5e-5, epochs = 2):
    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(model.parameters(), 
                                lr = 5e-5,
                                eps = 1e-08)
    
    
    # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
    
    training_results=[]

    for _ in trange(epochs, desc = 'Epoch'):
        
        # ========== Training ==========
        
        # Set model to training mode
        model.train()
        
        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch
            optimizer.zero_grad()
            # Forward pass
            train_output = model(b_input_ids, 
                                token_type_ids = None, 
                                attention_mask = b_input_mask, 
                                labels = b_labels)
            # Backward pass
            train_output.loss.backward()
            optimizer.step()
            # Update tracking variables
            tr_loss += train_output.loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        accuracy, precision, recall, specificity = evaluate(device, model, validation_dataloader)
        loss = tr_loss / nb_tr_steps

        print('\n\t - Train loss: {:.4f}'.format(loss))
        print('\t - Validation Precision: {:.4f}'.format(precision)) 
        print('\t - Validation Recall: {:.4f}'.format(recall) )
        print('\t - Validation Specificity: {:.4f}\n'.format(specificity))

        training_results.append({
            'loss':'{:.4f}'.format(loss),
            'accuracy':'{:.4f}'.format(accuracy),
            'precision':'{:.4f}'.format(precision),
            'recall':'{:.4f}'.format(recall),
            'specificity':'{:.4f}'.format(specificity)
        })

    return training_results
        
def evaluate(device, model, validation_dataloader):
   # ========== Validation ==========

        # Set model to evaluation mode
        model.eval()

        # Tracking variables
            
        val_accuracy = []
        val_precision = []
        val_recall = []
        val_specificity = []

        for batch in validation_dataloader:
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask, b_labels = batch

            with torch.no_grad():
                # Forward pass
                eval_output = model(b_input_ids, 
                                    token_type_ids = None, 
                                    attention_mask = b_input_mask)
                
            logits = eval_output.logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()
            # Calculate validation metrics
            b_accuracy, b_precision, b_recall, b_specificity = b_metrics(logits, label_ids)
            val_accuracy.append(b_accuracy)

            # Update precision only when (tp + fp) !=0; ignore nan
            if b_precision != 'nan': 
               val_precision.append(b_precision)

            # Update recall only when (tp + fn) !=0; ignore nan
            if b_recall != 'nan': 
               val_recall.append(b_recall)

            # Update specificity only when (tn + fp) !=0; ignore nan
            if b_specificity != 'nan': 
               val_specificity.append(b_specificity)

        accuracy = sum(val_accuracy)/len(val_accuracy)
        precision = sum(val_precision)/len(val_precision)
        recall = sum(val_recall)/len(val_recall)
        specificity = sum(val_specificity)/len(val_specificity)

        return (accuracy, precision, recall, specificity)