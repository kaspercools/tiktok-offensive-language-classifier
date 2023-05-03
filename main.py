import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tiktok_text_processing
import ml_utils
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import sys, getopt
from tabulate import tabulate
import random

max_F1 = 0
def print_rand_sentence(tokenizer: BertTokenizer, comments: list)-> None:
  '''Displays the tokens and respective IDs of a random text sample'''
  index = random.randint(0, len(comments)-1)
  table = np.array([tokenizer.tokenize(comments[index]), 
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comments[index]))]).T
  print(tabulate(table,
                 headers = ['Tokens', 'Token IDs'],
                 tablefmt = 'fancy_grid'))

def print_rand_sentence_encoding(tokenizer: BertTokenizer, item_count: int, token_ids: list, attention_masks:list) -> None:
  '''Displays tokens, token IDs and attention mask of a random text sample'''
  index = random.randint(0, item_count - 1)
  tokens = tokenizer.tokenize(tokenizer.decode(token_ids[index]))
  token_ids = [i.numpy() for i in token_ids[index]]
  attention = [i.numpy() for i in attention_masks[index]]

  table = np.array([tokens, token_ids, attention]).T
  print(tabulate(table, 
                 headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                 tablefmt = 'fancy_grid'))

def generate_plots(df: pd.DataFrame) -> None:
    df.groupby(['offensive']).size().plot.bar()
    plt.show()
    df.head()

def start_training_sequence(sourcefile: str, outputdir: str, dataFolder:str, learning_rate:float, val_ratio:float, batch_size:int, epochs:int, include_slang:bool=False, include_emoji:bool=False) -> None:
    global max_F1
    # read data from file
    df = ml_utils.read_data_frame(sourcefile)
    #preproccessing
    df['comment'] = df['comment'].apply(ml_utils.remove_links)
    if include_emoji:
        df['comment'] = df['comment'].apply(tiktok_text_processing.replace_emoji_w_token)

    #print(df.sample(25))
    # get comments and labels
    comments = df.comment.values    
    
    #generate_plots(df)
    
    # data exploration
    df['comment_len']  = df['comment'].str.len()
    
    ax = df.groupby(['comment_len']).size().plot.bar()
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 5) != 0:
            t.set_visible(False)

    #plt.show()

    comment_mean = df['comment'].str.len().mean()
    print('Average string length:', df['comment'].str.len().mean())
    print('Max sentence length: ', max([len(comment.split(' ')) for comment in comments]))

    
    tokenizer = ml_utils.generate_tokenizer(include_slang, include_emoji)
    token_id, attention_masks = ml_utils.encode_data(tokenizer, df['comment'])

    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)    

    labels = torch.tensor(df.offensive.values)

    #print_rand_sentence_encoding(tokenizer, len(comments), token_id, attention_masks)

    train_dataloader, validation_dataloader= ml_utils.create_datasets(token_id, attention_masks, labels, val_ratio, batch_size)

    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )       
    model.resize_token_embeddings(len(tokenizer))

    # Run on GPU
    model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_results = ml_utils.train(device, model, train_dataloader, validation_dataloader, learning_rate, epochs)

    comment_max_len = 150
    
    # add some custom validation
    new_comment = 'LOLOLOL @babaaibrahim the bitch was driving it. God damn I\'m not sexist but that is not a car that should be driven by a female ever lolike him and GOP needs CO to get to 270.'
    prediction = test_model_single_sample(device, model, tokenizer, new_comment, comment_max_len)
    print(prediction)
    new_comment = "cap"
    prediction = test_model_single_sample(device, model, tokenizer, new_comment, comment_max_len)
    print(prediction)
    
    timestr = time.strftime("%Y%m%d_%H%M%S")
    filepath = outputdir+'/'+timestr+'_offensive.bak'

    new_max_f1 = max([float(d['F1']) for d in training_results])

    print('F1 score '+ str(max_F1))

    is_new_max = new_max_f1 >= 0.86
    if is_new_max:
        max_F1 = new_max_f1

    print('F1 score '+ str(max_F1))
    #print(training_results)
    
    training_results.append({
        'lr':learning_rate,
        'batch_size': batch_size,
        'epochs':epochs
    })
    if is_new_max:
        ml_utils.save_model(model, training_results, filepath, not is_new_max)
    
def test_model_single_sample(device:torch.device, model:BertForSequenceClassification, tokenizer:BertTokenizer, new_comment:str, comments_max_len:int)->str:
    # We need Token IDs and Attention Mask for inference on the new sentence
    test_ids = []
    test_attention_mask = []

    # Apply the tokenizer
    encoding = ml_utils.preprocessing(new_comment, tokenizer, comments_max_len)

    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim = 0)
    test_attention_mask = torch.cat(test_attention_mask, dim = 0)

    # Forward pass, calculate logit predictions
    with torch.no_grad():
        output = model(test_ids.to(device), token_type_ids = None, attention_mask = test_attention_mask.to(device))

    prediction = 'Offensive' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'Not offensive'
    return prediction

def main(argv):
    batch_size = 32
    learning_rate = 5e-5
    val_ratio = 0.2
    epochs = 2
    outputdir='models'
    dataFolder='data'
    iterations=100

    opts, args = getopt.getopt(argv,"hi:o:lr:vr:e:b:n:",["ifile=","odir=", "learning_rate=", "validation_ratio=", "epochs=","batch_size=", "iterations="])
    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -i <inputfile> -o <outputdir> -lr <learning rate> -vr <validation ratio> -e <epochs> -b <batch size> -n <number of iterations>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--odir"):
            outputdir = arg
        elif opt in ("-lr", "--learning_rate"):
            learning_rate = float(arg)
        elif opt in ("-vr", "--validation_ratio"):
            val_ratio = float(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-n", "--iterations"):
            iterations = int(arg)
        elif opt in ("-d", "--data_folder"):
            dataFolder = int(arg)

    
    for i in range(iterations):
        start_training_sequence(inputfile, outputdir, dataFolder, learning_rate, val_ratio, batch_size, epochs, True, True)

if __name__ == '__main__':
    main(sys.argv[1:])
