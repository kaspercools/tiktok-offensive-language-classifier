import getopt
import random
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from tabulate import tabulate
from transformers import BertTokenizer, BertForSequenceClassification

from ou_ml import tiktok_text_processing, utils as ml_utils
from ou_ml.ml_metric import MlMetric
from ou_ml.tiktok_bert import TikTokBertBinaryClassifier

F_SCORE_THRESHOLD = 0.6


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


def train_sequence(source_file: str, zoomer_slang_file: str, learning_rate: float, adam_epsilon: float,
                   val_ratio: float,
                   batch_size: int, max_token_len: int, epochs: int, include_slang: bool = False,
                   include_emoji: bool = False) -> tuple[[dict], BertForSequenceClassification]:
    # read data from file
    df = ml_utils.read_data_frame(source_file)

    # preprocessing
    preprocess_comments(df, include_emoji)
    print_data_len(df)

    bert_classifier = initialize_classifier(adam_epsilon, batch_size, epochs, include_emoji, include_slang,
                                            learning_rate, max_token_len, zoomer_slang_file)

    token_id, attention_masks = bert_classifier.encode_data(df['comment'])
    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df.offensive.values)

    train_dataloader, validation_dataloader = ml_utils.create_datasets(token_id, attention_masks, labels, val_ratio,
                                                                       batch_size)

    if torch.cuda.is_available():
        bert_classifier.use_cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    training_results = ml_utils.train(device, bert_classifier, train_dataloader, validation_dataloader)

    sample_testing(bert_classifier, device)

    return training_results, bert_classifier.model


def initialize_classifier(adam_epsilon, batch_size, epochs, include_emoji, include_slang, learning_rate, max_token_len,
                          zoomer_slang_file) -> TikTokBertBinaryClassifier:
    bert_classifier = TikTokBertBinaryClassifier(include_slang, include_emoji, batch_size, max_token_len, learning_rate,
                                                 epochs,
                                                 adam_epsilon)
    try:
        bert_classifier.custom_voc_file = zoomer_slang_file
        bert_classifier.init_tokenizer()
    except Exception as ex:
        print("Error while initializing tokenizer: ", ex)
    return bert_classifier


def print_data_len(df: DataFrame) -> None:
    df['comment_len'] = df['comment'].str.len()
    comment_mean = df['comment'].str.len().mean()
    print('Average string length:', comment_mean)
    print('Max sentence length: ', max([len(comment.split(' ')) for comment in df['comment']]))


def sample_testing(classifier: TikTokBertBinaryClassifier, device: torch.device) -> None:
    # add some custom validation
    new_comment = 'LOLOLOL @babaaibrahim the bitch was driving it. God damn I\'m not sexist but that is not a car ' \
                  'that should be driven by a female ever lolike him and GOP needs CO to get to 270. '
    prediction = test_model_single_sample(device, classifier, new_comment)
    print(prediction)
    new_comment = "cap"
    prediction = test_model_single_sample(device, classifier, new_comment)
    print(prediction)


def preprocess_comments(df: DataFrame, include_emoji: bool) -> None:
    df['comment'] = df['comment'].apply(ml_utils.remove_links)
    if include_emoji:
        df['comment'] = df['comment'].apply(tiktok_text_processing.replace_emoji_w_token)


def test_model_single_sample(device: torch.device, classifier: TikTokBertBinaryClassifier, comment: str) -> str:
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
    custom_voc_file = None
    input_file = None
    include_emoji = False
    include_slang = False

    opts, args = getopt.getopt(argv, "hi:o:l;a:v:e:b:t:n:c:m",
                               [
                                   "help"
                                   "ifile=",
                                   "outdir=",
                                   "learning_rate=",
                                   "validation_ratio=",
                                   "epochs=",
                                   "batch_size=",
                                   "max_token_len=",
                                   "iterations=",
                                   "custom_voc=",
                                   "slang=",
                                   "emoji="
                               ])

    for opt, arg in opts:
        if opt == '-h':
            print(
                'main.py -i <input file> -o <output dir> -l <learning rate> -a <adam_epsilon> -v <validation ratio> '
                '-e <epochs> -b <batch size> -t <max token len> -n <number of iterations> -c <custom vocabulary input '
                'file> -m (flag that enables emoji tokenization)')
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
        elif opt in ("-c", "--custom_voc"):
            custom_voc_file = arg
            include_slang = True
        elif opt in ('-m', "--emoji"):
            include_emoji = True

    if not input_file:
        # if no inputfile, let the user know:
        print("No input provided! Please provide a csv file with 2 columns: 1) comment, and 2) offensive (binary "
              "label)")
        sys.exit()
    elif include_slang and not custom_voc_file:
        print(TikTokBertBinaryClassifier.INCLUDE_SLANG_ERR_MSG)
        sys.exit()

    for i in range(iterations):
        training_results, model = train_sequence(input_file, custom_voc_file, learning_rate, adam_epsilon, val_ratio,
                                                 batch_size, max_token_len, epochs,
                                                 include_slang, include_emoji)

        new_max_f_score = max([float(d['F1']) for d in training_results])
        print('F1 score ' + str(max_f_score))

        is_new_max = new_max_f_score >= F_SCORE_THRESHOLD
        if is_new_max:
            max_f_score = new_max_f_score

        export_data(batch_size, epochs, is_new_max, learning_rate, model, output_dir, training_results)


def export_data(batch_size: int, epochs: int, is_new_max: bool, learning_rate: float,
                model: BertForSequenceClassification, output_dir: str, training_results: MlMetric) -> None:
    time_str = time.strftime("%Y%m%d_%H%M%S")
    filepath = output_dir + '/' + time_str + '_offensive.bak'
    training_results.append({
        'lr': learning_rate,
        'batch_size': batch_size,
        'epochs': epochs
    })
    if is_new_max:
        ml_utils.save_model(model, training_results, filepath, not is_new_max)


if __name__ == '__main__':
    main(sys.argv[1:])
