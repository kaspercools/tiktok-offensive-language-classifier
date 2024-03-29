import gc
import json
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tqdm import trange
from transformers import BertForSequenceClassification

from . import MlMetric
from . import TikTokBertBinaryClassifier

#https://towardsdatascience.com/confusion-matrix-and-class-statistics-68b79f4f510b
def b_tp(preds, labels) -> float:
    '''Returns True Positives (TP): count of correct predictions of actual class 1'''
    return sum([preds == labels and preds == 1 for preds, labels in zip(preds, labels)])


def b_fp(preds, labels) -> float:
    '''Returns False Positives (FP): count of wrong predictions of actual class 1'''
    return sum([preds != labels and preds == 1 for preds, labels in zip(preds, labels)])


def b_tn(preds, labels) -> float:
    '''Returns True Negatives (TN): count of correct predictions of actual class 0'''
    return sum([preds == labels and preds == 0 for preds, labels in zip(preds, labels)])


def b_fn(preds, labels) -> float:
    '''Returns False Negatives (FN): count of wrong predictions of actual class 0'''
    return sum([preds != labels and preds == 0 for preds, labels in zip(preds, labels)])


def calculate_ml_metrics(predictions, labels, reference_axis=1) -> MlMetric:
    '''
  Returns the following metrics:
    - accuracy    = (TP + TN) / N
    - precision   = TP / (TP + FP)
    - recall      = TP / (TP + FN)
    - specificity = TN / (TN + FP)
  '''
    if reference_axis:
        predictions = np.argmax(predictions, axis=reference_axis)

    predictions = predictions.flatten()
    labels = labels.flatten()

    tp = b_tp(predictions, labels)
    tn = b_tn(predictions, labels)
    fp = b_fp(predictions, labels)
    fn = b_fn(predictions, labels)

    b_accuracy = (tp + tn) / len(labels)
    b_precision = tp / (tp + fp) if (tp + fp) > 0 else 'nan'
    b_recall = tp / (tp + fn) if (tp + fn) > 0 else 'nan'
    b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 'nan'
    return MlMetric(b_accuracy, b_precision, b_recall, b_specificity)


def clear_gpu_cache() -> None:
    # clear cache on re-run
    gc.collect()
    torch.cuda.empty_cache()


def create_datasets(token_ids: list, attention_masks: list, labels: list, val_ratio: float,
                    batch_size: int = 16) -> tuple[DataLoader, DataLoader]:
    # Recommended batch size: 16, 32. See: https://arxiv.org/pdf/1810.04805.pdf

    # Indices of the train and validation splits stratified by labels
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=val_ratio,
        shuffle=True,
        stratify=labels)

    # Train and validation sets
    train_set = TensorDataset(token_ids[train_idx],
                              attention_masks[train_idx],
                              labels[train_idx])

    val_set = TensorDataset(token_ids[val_idx],
                            attention_masks[val_idx],
                            labels[val_idx])

    # Prepare DataLoader
    train_dataloader = DataLoader(
        train_set,
        sampler=RandomSampler(train_set),
        batch_size=batch_size
    )

    validation_dataloader = DataLoader(
        val_set,
        sampler=SequentialSampler(val_set),
        batch_size=batch_size
    )

    return train_dataloader, validation_dataloader


def read_data_frame(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.head()
    return df


def remove_links(comment: str) -> str:
    comment = re.sub(r'http\S+', '', comment)
    return comment


def remove_usernames(comment: str) -> str:
    comment = re.sub(r'@\S+', '', comment)
    return comment


def save_model(model: BertForSequenceClassification, training_results: list, filepath: str,
               only_json: bool = False) -> None:
    if not only_json:
        torch.save(model.state_dict(), filepath)

    # Serializing json
    json_object = json.dumps(training_results, indent=4)

    # Writing to sample.json
    with open(filepath + '_results.json', "w") as outfile:
        outfile.write(json_object)


def train(device: torch.device, classifier: TikTokBertBinaryClassifier, train_dataloader: DataLoader,
          validation_dataloader: DataLoader) -> [dict]:
    # Recommended learning rates (Adam): 5e-5, 3e-5, 2e-5. See: https://arxiv.org/pdf/1810.04805.pdf
    optimizer = torch.optim.AdamW(classifier.model.parameters(),
                                  lr=classifier.learning_rate,
                                  eps=classifier.adam_epsilon)

    # Recommended number of epochs: 2, 3, 4. See: https://arxiv.org/pdf/1810.04805.pdf
    training_results = []

    for _ in trange(classifier.epochs, desc='Epoch'):
        # ========== Training ==========
        # Set model to training mode
        classifier.model.train()

        # Tracking variables
        tr_loss = 0

        nb_tr_steps, tr_loss = train_single_epoch(classifier, device, optimizer, tr_loss,
                                                  train_dataloader)

        ml_metrics = evaluate(device, classifier, validation_dataloader)
        ml_metrics.loss = tr_loss / nb_tr_steps
        training_result = training_result_to_dict(ml_metrics)

        print_training_result(training_result)

        training_results.append(training_result)

    return training_results


def train_single_epoch(classifier, device, optimizer, tr_loss, train_dataloader) \
        -> tuple[int, int]:
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_mask, b_labels = batch
        optimizer.zero_grad()
        # Forward pass
        train_output = classifier.model(b_input_ids,
                                        token_type_ids=None,
                                        attention_mask=b_input_mask,
                                        labels=b_labels)
        # Backward pass
        train_output.loss.backward()
        optimizer.step()

        # Update tracking variables
        tr_loss += train_output.loss.item()
        nb_tr_examples += b_input_ids.size(0)
        nb_tr_steps += 1

    return nb_tr_steps, tr_loss


def training_result_to_dict(ml_metrics: MlMetric) -> dict:
    return {
        'F1': '{:.4f}'.format(ml_metrics.calculate_f_score()),
        'loss': '{:.4f}'.format(ml_metrics.loss),
        'accuracy': '{:.4f}'.format(ml_metrics.accuracy),
        'precision': '{:.4f}'.format(ml_metrics.precision),
        'recall': '{:.4f}'.format(ml_metrics.recall),
        'specificity': '{:.4f}'.format(ml_metrics.specificity),
        'labels': ml_metrics.labels,
        'predictions': ml_metrics.predictions
    }


def print_training_result(training_result: dict) -> None:
    print('\t - {}'.format(training_result.get('precision')))
    print('\t - {}'.format(training_result.get('recall')))
    print('\t - {}\n'.format(training_result.get('specificity')))
    print('\t - {}\n'.format(training_result.get('F1')))


def evaluate(device: torch.device, classifier: TikTokBertBinaryClassifier,
             validation_dataloader: DataLoader) -> MlMetric:
    # Set model to evaluation mode
    classifier.model.eval()
    all_predictions = []
    all_label_ids = []

    # Tracking variables

    val_accuracy = []
    val_precision = []
    val_recall = []
    val_specificity = []

    for batch in validation_dataloader:
        b_labels, eval_output = evaluate_batch(batch, device, classifier.model)

        predictions = eval_output.logits.detach().cpu().numpy()

        label_ids = b_labels.to('cpu').numpy()
        # Calculate validation metrics
        mlMetric = calculate_ml_metrics(predictions, label_ids)

        val_accuracy.append(mlMetric.accuracy)

        # Update precision only when (tp + fp) !=0; ignore nan
        if mlMetric.precision != 'nan':
            val_precision.append(mlMetric.precision)

        # Update recall only when (tp + fn) !=0; ignore nan
        if mlMetric.recall != 'nan':
            val_recall.append(mlMetric.recall)

        # Update specificity only when (tn + fp) !=0; ignore nan
        if mlMetric.specificity != 'nan':
            val_specificity.append(mlMetric.specificity)

        all_predictions = all_predictions + predictions.tolist()
        all_label_ids = all_label_ids + label_ids.tolist()

    accuracy = sum(val_accuracy) / len(val_accuracy)
    precision = sum(val_precision) / len(val_precision)
    recall = sum(val_recall) / len(val_recall)
    specificity = sum(val_specificity) / len(val_specificity)

    return MlMetric(accuracy, precision, recall, specificity, all_predictions, all_label_ids)


def evaluate_batch(batch: list, device: torch.device, model: BertForSequenceClassification) -> tuple[list, list]:
    b_input_ids, b_input_mask, b_labels = tuple(t.to(device) for t in batch)
    with torch.no_grad():
        # Forward pass
        eval_output = model(b_input_ids,
                            token_type_ids=None,
                            attention_mask=b_input_mask)
    return b_labels, eval_output
