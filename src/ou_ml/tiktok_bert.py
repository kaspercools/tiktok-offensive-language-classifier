import pandas as pd
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

from . import tiktok_text_processing


class TikTokBertBinaryClassifier:
    INCLUDE_SLANG_ERR_MSG = 'You seem to be missing a slang dataset (include_slang=True). Please configure the slang dataset '
    'location before calling the tokenizer initializer'
    def __init__(self, include_slang: bool, include_emoji: bool,
                 batch_size: float = 32,
                 tokens_max_len: int = 150,
                 learning_rate: float = 5e-5,
                 epochs: int = 2,
                 adam_epsilon=1e-08
                 # https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/
                 ):
        self.include_slang = include_slang
        self.include_emoji = include_emoji

        self.tokenizer = None
        self.custom_voc_file = None
        self.label_column = 'Offensive'

        self.tokens_max_len = tokens_max_len
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=2,
            output_attentions=False,
            output_hidden_states=False,
        )

        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.adam_epsilon = adam_epsilon

    def encode_data(self, items: DataLoader) -> tuple[list, list]:
        token_ids = []
        attention_masks = []

        for sample in items:
            encoding_dict = self.preprocessing(sample.lower())
            token_ids.append(encoding_dict['input_ids'])
            attention_masks.append(encoding_dict['attention_mask'])

        return token_ids, attention_masks

    def get_slang(self) -> list:
        df = pd.read_csv(self.custom_voc_file)
        df['keyword'] = df['keyword'].apply(lambda c: c.lower())
        df.head()
        return df.keyword.to_list()

    def preprocessing(self, input_text: str) -> BertTokenizer:
        '''
      Returns <class transformers.tokenization_utils_base.BatchEncoding> with the following fields:
        - input_ids: list of token ids
        - token_type_ids: list of token type ids
        - attention_mask: list of indices (0,1) specifying which tokens should considered by the model (return_attention_mask = True).
      '''
        # https://github.com/huggingface/transformers/issues/1490
        return self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.tokens_max_len,
            pad_to_max_length=True,
            return_attention_mask=True,
            truncation=True,
            return_tensors='pt'
        )

    def use_cuda(self) -> None:
        self.model.cuda()

    def init_tokenizer(self):

        if self.include_slang and not self.custom_voc_file:
            raise Exception(TikTokBertBinaryClassifier.INCLUDE_SLANG_ERR_MSG)

        self.tokenizer = self.__generate_tokenizer(self.include_slang, self.include_emoji)
        self.model.resize_token_embeddings(len(self.tokenizer))

    def __generate_tokenizer(self, include_slang: bool = False, include_emoji: bool = False) -> BertTokenizer:
        tokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased',
            do_lower_case=True)

        # add custom tik~tok gen-z tokens to tokenizer
        if include_slang:
            num_genz_slang_added_tokens = tokenizer.add_tokens(self.get_slang())
        if include_emoji:
            num_emoji_added_tokens = tokenizer.add_tokens(
                tiktok_text_processing.get_emoji_tokens())  # cfr https://arxiv.org/pdf/1910.13793.pdf

        return tokenizer
