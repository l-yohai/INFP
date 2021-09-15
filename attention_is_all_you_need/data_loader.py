import spacy
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator


class Multi30kDataLoader():
    def __init__(self, batch_size, device):
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

        self.SOURCE = Field(tokenize=self.tokenize_deutsch, init_token='<sos>',
                            eos_token='<eos>', lower=True, batch_first=True)
        self.TARGET = Field(tokenize=self.tokenize_english, init_token='<sos>',
                            eos_token='<eos>', lower=True, batch_first=True)

        self.train_data, self.valid_data, self.test_data = Multi30k.splits(
            root='./data', exts=('.de', '.en'), fields=(self.SOURCE, self.TARGET))

        self.SOURCE.build_vocab(self.train_data, min_freq=2)
        self.TARGET.build_vocab(self.train_data, min_freq=2)

        self.train_loader, self.valid_loader, self.test_loader = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data), batch_size=batch_size, device=device)

        # self.source_padding_token_index = self.SOURCE.vocab.stoi['<pad>']
        # self.target_padding_token_index = self.TARGET.vocab.stoi['<pad>']
        # self.target_start_of_sentence_token_index = self.TARGET.vocab.stoi['<sos>']
        self.source_vocab_size = len(self.SOURCE.vocab)
        self.target_vocab_size = len(self.TARGET.vocab)

    def tokenize_deutsch(self, text):
        return [token.text for token in self.spacy_de.tokenizer(text)]

    def tokenize_english(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]

    def get_train_loader(self):
        return self.train_loader

    def get_valid_loader(self):
        return self.valid_loader

    def get_test_loader(self):
        return self.test_loader
