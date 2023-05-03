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

from tiktok_bert import TikTokBertClassifier

CURRENT_F_SCORE_THRESHOLD = 0.6


def print_rand_sentence(tokenizer: BertTokenizer, comments: list) -> None:
    """Displays the tokens and respective IDs of a random text sample"""
    index = random.randint(0, len(comments) - 1)
    table = np.array([tokenizer.tokenize(comments[index]),
                      tokenizer.convert_tokens_to_ids(tokenizer.tokenize(comments[index]))]).T
    print(tabulate(table,
                   headers=['Tokens', 'Token IDs'],
                   tablefmt='fancy_grid'))


def print_rand_sentence_encoding(tokenizer: BertTokenizer, item_count: int, token_ids: list,
                                 attention_masks: list) -> None:
    """Displays tokens, token IDs and attention mask of a random text sample"""
    index = random.randint(0, item_count - 1)
    tokens = tokenizer.tokenize(tokenizer.decode(token_ids[index]))
    token_ids = [i.numpy() for i in token_ids[index]]
    attention = [i.numpy() for i in attention_masks[index]]

    table = np.array([tokens, token_ids, attention]).T
    print(tabulate(table,
                   headers=['Tokens', 'Token IDs', 'Attention Mask'],
                   tablefmt='fancy_grid'))


def generate_plots(df: pd.DataFrame) -> None:
    # show plot grouped by comment len
    ax = df.groupby(['comment_len']).size().plot.bar()
    for i, t in enumerate(ax.get_xticklabels()):
        if (i % 5) != 0:
            t.set_visible(False)

    plt.show()


def train_sequence(sourcefile: str, learning_rate: float, adam_epsilon: float, val_ratio: float,
                   batch_size: int, max_token_len: int, epochs: int, include_slang: bool = False,
                   include_emoji: bool = False) -> tuple[[dict], BertForSequenceClassification]:
    # read data from file
    df = ml_utils.read_data_frame(sourcefile)

    # preprocessing
    preprocess_comments(df, include_emoji)
    print_data_len(df)

    classifier = TikTokBertClassifier(include_slang, include_emoji, batch_size, max_token_len, learning_rate, epochs,
                                      adam_epsilon)

    token_id, attention_masks = classifier.encode_data(df['comment'])
    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df.offensive.values)

    train_dataloader, validation_dataloader = ml_utils.create_datasets(token_id, attention_masks, labels, val_ratio,
                                                                       batch_size)

    if torch.cuda.is_available():
        classifier.use_cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    training_results = ml_utils.train(device, classifier, train_dataloader, validation_dataloader)

    sample_testing(classifier, device)

    return training_results, classifier.model


def print_data_len(df):
    df['comment_len'] = df['comment'].str.len()
    comment_mean = df['comment'].str.len().mean()
    print('Average string length:', comment_mean)
    print('Max sentence length: ', max([len(comment.split(' ')) for comment in df['comment']]))


def sample_testing(classifier, device):
    # add some custom validation
    new_comment = 'LOLOLOL @babaaibrahim the bitch was driving it. God damn I\'m not sexist but that is not a car ' \
                  'that should be driven by a female ever lolike him and GOP needs CO to get to 270. '
    prediction = test_model_single_sample(device, classifier, new_comment)
    print(prediction)
    new_comment = "cap"
    prediction = test_model_single_sample(device, classifier, new_comment)
    print(prediction)


def preprocess_comments(df, include_emoji):
    df['comment'] = df['comment'].apply(ml_utils.remove_links)
    if include_emoji:
        df['comment'] = df['comment'].apply(tiktok_text_processing.replace_emoji_w_token)


def test_model_single_sample(device: torch.device, classifier: TikTokBertClassifier, comment: str) -> str:
    # We need Token IDs and Attention Mask for inference on the new sentence
    test_ids = []
    test_attention_mask = []

    # Apply the tokenizer
    encoding = ml_utils.preprocessing(comment, classifier.tokenizer, classifier.tokens_max_len)

    # Extract IDs and Attention Mask
    test_ids.append(encoding['input_ids'])
    test_attention_mask.append(encoding['attention_mask'])
    test_ids = torch.cat(test_ids, dim=0)
    test_attention_mask = torch.cat(test_attention_mask, dim=0)

    # Forward pass, calculate logit predictions
    with torch.no_grad():
        output = classifier.model(test_ids.to(device), token_type_ids=None,
                                  attention_mask=test_attention_mask.to(device))

    prediction = 'Offensive' if np.argmax(output.logits.cpu().numpy()).flatten().item() == 1 else 'Not offensive'
    return prediction


def main(argv):
    max_f_score = 0
    batch_size = 32
    learning_rate = 5e-5
    adam_epsilon = 1e-08
    val_ratio = 0.2
    epochs = 2
    output_dir = 'models'
    iterations = 100
    max_token_len = 150
    input_file = None

    opts, args = getopt.getopt(argv, "h:i:o:l;a:v:e:b:t:n:",
                               [
                                   "help"
                                   "ifile=",
                                   "outdir=",
                                   "learning_rate=",
                                   "validation_ratio=",
                                   "epochs=",
                                   "batch_size=",
                                   "max_token_len=",
                                   "iterations="])
    print(opts)
    for opt, arg in opts:
        if opt == '-h':
            print(
                'main.py -i <input file> -o <output dir> -l <learning rate> -a <adam_epsilon> -v <validation ratio> '
                '-e <epochs> -b <batch size> -t <max token len> -n <number of iterations>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            input_file = arg
        elif opt in ("-o", "--odir"):
            output_dir = arg
        elif opt in ("-l", "--learning_rate"):
            learning_rate = float(arg)
        elif opt in ("-a", "--adam_epsilon"):
            adam_epsilon = float(arg)
        elif opt in ("-v", "--validation_ratio"):
            val_ratio = float(arg)
        elif opt in ("-e", "--epochs"):
            epochs = int(arg)
        elif opt in ("-b", "--batch_size"):
            batch_size = int(arg)
        elif opt in ("-n", "--iterations"):
            iterations = int(arg)
        elif opt in ("-t", "--max_token_len"):
            max_token_len = int(arg)

    if not input_file:
        # if no inputfile, let the user know:
        print("No input provided! Please provide a csv file with 2 columns: 1) comment, and 2) offensive (binary "
              "label)")
        quit()

    for i in range(iterations):
        training_results, model = train_sequence(input_file, learning_rate, adam_epsilon, val_ratio,
                                                 batch_size, max_token_len, epochs,
                                                 True, True)
        new_max_f_score = max([float(d['F1']) for d in training_results])
        print('F1 score ' + str(max_f_score))

        is_new_max = new_max_f_score >= CURRENT_F_SCORE_THRESHOLD
        if is_new_max:
            max_f_score = new_max_f_score

        export_data(batch_size, epochs, is_new_max, learning_rate, model, output_dir, training_results)


def export_data(batch_size, epochs, is_new_max, learning_rate, model, outputdir, training_results):
    time_str = time.strftime("%Y%m%d_%H%M%S")
    filepath = outputdir + '/' + time_str + '_offensive.bak'
    training_results.append({
        'lr': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    })
    if is_new_max:
        ml_utils.save_model(model, training_results, filepath, not is_new_max)


if __name__ == '__main__':
    main(sys.argv[1:])
