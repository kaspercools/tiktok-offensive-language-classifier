import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tiktok_text_processing
import ml_utils
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import sys, getopt
from tabulate import tabulate
import random

def print_rand_sentence(tokenizer, comments):
  '''Displays the tokens and respective IDs of a random text sample'''
  index = random.randint(0, len(comments)-1)
  table = np.array([tokenizer.tokenize(comments[index]), 
                    tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comments[index]))]).T
  print(tabulate(table,
                 headers = ['Tokens', 'Token IDs'],
                 tablefmt = 'fancy_grid'))

def print_rand_sentence_encoding(tokenizer, item_count, token_id, attention_masks):
  '''Displays tokens, token IDs and attention mask of a random text sample'''
  index = random.randint(0, item_count - 1)
  tokens = tokenizer.tokenize(tokenizer.decode(token_id[index]))
  token_ids = [i.numpy() for i in token_id[index]]
  attention = [i.numpy() for i in attention_masks[index]]

  table = np.array([tokens, token_ids, attention]).T
  print(tabulate(table, 
                 headers = ['Tokens', 'Token IDs', 'Attention Mask'],
                 tablefmt = 'fancy_grid'))

def generate_plots(df):
    df.groupby(['offensive']).size().plot.bar()
    plt.show()
    df.head()

def start_training_sequence(sourcefile, outputdir, learning_rate, val_ratio, batch_size, epochs):
    # read data from file
    df = ml_utils.read_data_frame(sourcefile)
    #preproccessing
    # data preperation
    df['comment'] = df['comment'].apply(ml_utils.remove_links)
    # data preperation
    token_dict = tiktok_text_processing.get_emoji_token_dic()
    df['comment'] = df['comment'].apply(lambda comment: tiktok_text_processing.replace_emoji_w_token(comment, token_dict))
    
    print(df.sample(25))
    # get comments and labels
    comments = df.comment.values
    
    #generate_plots(df)

    tokenizer = BertTokenizer.from_pretrained(
        'bert-base-uncased',
        do_lower_case = True
    )
    
    # data exploration
    print_rand_sentence(tokenizer, comments)

    print('Max sentence length: ', max([len(comment) for comment in comments]))

    token_id, attention_masks= ml_utils.encode_data(tokenizer, comments)

    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df.offensive.values)

    print_rand_sentence_encoding(tokenizer, len(comments), token_id, attention_masks)

    train_dataloader, validation_dataloader= ml_utils.create_datasets(token_id, attention_masks, labels, val_ratio, batch_size)

    # Load the BertForSequenceClassification model
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels = 2,
        output_attentions = False,
        output_hidden_states = False,
    )       

    # add custom tiktok gen-z tokens to tokenizer
    num_genz_slang_added_tokens = tokenizer.add_tokens(tiktok_text_processing.get_genz_slang())
    num_emoji_added_tokens = tokenizer.add_tokens(tiktok_text_processing.get_emoji_tokens()) # cfr https://arxiv.org/pdf/1910.13793.pdf
    # call resesize_token_embeddings and pass new token len
    model.resize_token_embeddings(len(tokenizer))

    # Run on GPU
    model.cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    training_results = ml_utils.train(device, model, train_dataloader, validation_dataloader, learning_rate, epochs)
    print(training_results)

    comment_max_len = 512#np.max([len(sample) for sample in comments])
    
    # add some custom validation
    new_comment = 'How do you make a dead coon float? Take your foot off its head andlet it rise to the surface asshole'
    prediction = test_model_single_sample(device, model, tokenizer, new_comment, comment_max_len)
    new_sentence = "cap"

    prediction = test_model_single_sample(device, model, tokenizer, new_comment, comment_max_len)

    timestr = time.strftime("%Y%m%d_%H%M%S")
    filepath = outputdir+'/'+timestr+'_offensive.bak'
    ml_utils.save_model(model, training_results, filepath)

    
def test_model_single_sample(device, model, tokenizer, new_comment, comments_max_len):
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
    batch_size =4
    learning_rate = 5e-5
    val_ratio = 0.2
    epochs = 2
    outputdir='models'
    iterations=1

    opts, args = getopt.getopt(argv,"hi:o:lr:vr:e:b:x:",["ifile=","odir=", "learning_rate=", "validation_ratio=", "epochs=","batch_size="])
    for opt, arg in opts:
        if opt == '-h':
            print ('main.py -i <inputfile> -o <outputdir> -lr <learning rate> -vr <validation ratio> -e <epochs> -b <batch size> -it <number of iterations>')
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
        elif opt in ("-x", "--iterations"):
            iterations = int(arg)

    df = ml_utils.read_data_frame(inputfile)
    #preproccessing
    
    for i in range(iterations):
        start_training_sequence(inputfile, outputdir, learning_rate, val_ratio, batch_size, epochs)

if __name__ == '__main__':
    main(sys.argv[1:])
