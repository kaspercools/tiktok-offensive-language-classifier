from transformers import BertForSequenceClassification
import ml_utils
from torch.utils.data import DataLoader


class TikTokBertClassifier:
    def __init__(self, include_slang: bool, include_emoji: bool, batch_size: float, learning_rate: float = 5e-5,
                 epochs: int = 2,
                 adam_epsilon=1e-08  # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
                 ):

        self.tokenizer = ml_utils.generate_tokenizer(include_slang, include_emoji)
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.adam_epsilon = adam_epsilon

    def encode_data(self, items: DataLoader):
        token_id, attention_masks = ml_utils.encode_data(self.tokenizer, items)
        return token_id, attention_masks

    def use_cuda(self):
        self.model.cuda()
