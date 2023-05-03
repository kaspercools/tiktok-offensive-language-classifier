import time

import pandas as pd
import torch
from pandas import DataFrame
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

import ml_utils
import tiktok_text_processing
from tiktok_bert import TikTokBertClassifier


def evaluate_samples(bert_classifier: TikTokBertClassifier, device: torch.cuda.device, df: DataFrame):
    df['full_text'] = df['full_text'].apply(tiktok_text_processing.replace_emoji_w_token)

    df['label'] = df['label'].apply(lambda x: 1 if x == "Offensive" else 0)

    token_id, attention_masks = bert_classifier.encode_data(df['full_text'])
    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    labels = torch.tensor(df.label.values.astype(int))
    val_set = TensorDataset(token_id, attention_masks, labels)

    validation_dataloader = DataLoader(
        val_set,
        sampler=SequentialSampler(val_set),
        batch_size=32
    )

    model_metrics = ml_utils.evaluate(device, bert_classifier,
                                      validation_dataloader)
    return ml_utils.training_result_to_dict(model_metrics)


def run(model_path: str) -> None:
    # Load the pre-trained model
    bert_classifier = TikTokBertClassifier(True, True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    bert_classifier.model.load_state_dict(torch.load(model_path, map_location=device))

    bert_classifier.model.eval()

    df = pd.read_csv('public_data_labeled.csv').sample(2000)
    training_result = (evaluate_samples(bert_classifier, device, df))
    ml_utils.print_training_result(training_result)
    time_str = time.strftime("%Y%m%d_%H%M%S")
    filepath = 'eval_result/' + time_str + '_eval'
    ml_utils.save_model(bert_classifier.model, [training_result], filepath, True)


if __name__ == '__main__':
    run('data/20230426_002347.bin')
