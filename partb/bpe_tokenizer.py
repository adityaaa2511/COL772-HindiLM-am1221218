import json
import re
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
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
        if isinstance(corpus, list):
            corpus = "\n".join(corpus)

        word_freqs = defaultdict(int)
        words = re.findall(r"\S+", corpus)
        for word in words:
            word_freqs[word + "</w>"] += 1 # Add </w> to indicate end of word for now

        splits = {}
        for word, freq in word_freqs.items():
            word_without_end = word[:-4]
            chars = list(word_without_end) + ["</w>"]
            splits[" ".join(chars)] = freq  # Add delimiter between characters and end of word

        char_vocab = set()
        for word in splits:
            for ch in word.split():
                char_vocab.add(ch)

        for cp in range(0x0900, 0x0980): # Add devnagiri characters to the character vocabulary
            char_vocab.add(chr(cp))

        current_vocab_size = len(self.special_tokens) + len(char_vocab)

        self.merges = []
        while current_vocab_size < self.target_vocab_size:
            stats = self.get_stats(splits)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 1:
                break
            splits = self.merge_pair(best_pair, splits)
            self.merges.append(best_pair)
            current_vocab_size += 1

        # Build the final vocabulary
        self.vocab = {}
        idx = 0
        for token in self.special_tokens:
            self.vocab[token] = idx
            idx += 1

        for ch in sorted(char_vocab):
            if ch not in self.vocab:
                self.vocab[ch] = idx
                idx += 1

        for pair_a, pair_b in self.merges:
            merged = pair_a + pair_b
            if merged not in self.vocab:
                self.vocab[merged] = idx
                idx += 1

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def apply_merges(self, token_list):
        for pair_a, pair_b in self.merges:
            i = 0
            while i < len(token_list) - 1:
                if token_list[i] == pair_a and token_list[i + 1] == pair_b:
                    token_list = token_list[:i] + [pair_a + pair_b] + token_list[i + 2:]
                    # Don't increment i so we can check for further merges at same position
                else:
                    i += 1
        return token_list

    def encode(self, text):
        unk_id = self.vocab.get("<UNK>", 1)
        words = re.findall(r"\S+", text)
        token_ids = []

        for word in words:
            chars = list(word) + ["</w>"]
            tokens = self.apply_merges(chars)
            for token in tokens:
                token_ids.append(self.vocab.get(token, unk_id))

        return token_ids

    def decode(self, token_ids):
        tokens = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, "<UNK>")
            tokens.append(token)

        text = "".join(tokens)
        text = text.replace("</w>", " ")
        return text.strip()

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