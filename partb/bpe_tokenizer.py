import json
import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}

    def get_stats(self, splits):
        counts = defaultdict(int)
        for word, freq in splits.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                counts[(symbols[i], symbols[i + 1])] += freq
        return counts

    def merge_pair(self, pair, splits):

        new_splits = {}
        bigram = re.escape(pair[0]) + r"\s+" + re.escape(pair[1])
        pattern = re.compile(bigram)
        replacement = pair[0] + pair[1]
        for word, freq in splits.items():
            new_word = pattern.sub(replacement, word)
            new_splits[new_word] = freq
        return new_splits

    def train(self, corpus):
        raise NotImplementedError("Training method not implemented yet.")

    def encode(self, text):
        raise NotImplementedError("Encoding method not implemented yet.")

    def decode(self, token_ids):
        raise NotImplementedError("Decoding method not implemented yet.")

    def save(self, dirpath):
        data = {
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        filepath = dirpath + "tokenizer.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.target_vocab_size = data["target_vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.vocab = data["vocab"]
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}

    def get_vocab_size(self):
        return len(self.vocab)

    def get_unk_id(self):
        return self.vocab.get("<UNK>", -1)