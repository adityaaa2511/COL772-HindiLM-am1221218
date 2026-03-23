import json
import re
import os
import time
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=10000, special_tokens=None):
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<|PAD|>", "<|UNK|>", "<|SOS|>", "<|EOS|>"]
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}
        self._special_token_set = set(self.special_tokens)
        escaped = sorted((re.escape(t) for t in self.special_tokens), key=len, reverse=True)
        self._special_pattern = re.compile("|".join(escaped)) if escaped else None

    def _segments(self, text):
        if not self._special_pattern:
            return [text]
        segments = []
        last = 0
        for m in self._special_pattern.finditer(text):
            s, e = m.span()
            if s > last:
                segments.append(text[last:s])
            segments.append(m.group())
            last = e
        if last < len(text):
            segments.append(text[last:])
        return segments

    def _split_words(self, text):
        return re.findall(r'\S+|\s+', text)

    def get_stats(self, splits):
        counts = defaultdict(int)
        for tokens, freq in splits.items():
            for i in range(len(tokens) - 1):
                counts[(tokens[i], tokens[i + 1])] += freq
        return counts

    def merge_pair(self, pair, splits):
        a, b = pair
        new_splits = defaultdict(int)
        for symbols, freq in splits.items():
            merged = []
            i = 0
            L = len(symbols)
            while i < L:
                if i < L-1 and symbols[i] == a and symbols[i+1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(symbols[i])
                    i += 1
            new_splits[tuple(merged)] += freq
        return new_splits

    def train(self, corpus, max_time_seconds=9000):
        segment_freqs = defaultdict(int)
        if isinstance(corpus, list):
            lines = corpus
        else:
            lines = corpus.splitlines()

        for line in lines:
            for segment in self._segments(line):
                if segment in self._special_token_set:
                    continue
                words = self._split_words(segment)
                for w in words:
                    if w:
                        segment_freqs[w] += 1

        splits = defaultdict(int)
        char_vocab = set()

        for word, freq in segment_freqs.items():
            chars = tuple(word)
            splits[chars] += freq
            char_vocab.update(chars)

        char_vocab.add(" ")

        current_vocab_size = len(self.special_tokens) + len(char_vocab)

        self.merges = []
        train_start = time.monotonic()
        while current_vocab_size < self.target_vocab_size:
            if time.monotonic() - train_start > max_time_seconds:  # Timeout 
                break

            stats = self.get_stats(splits)
            if not stats:
                break

            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2:  # Prevent overfitting to very rare pairs
                break
            splits = self.merge_pair(best_pair, splits)
            self.merges.append(best_pair)
            current_vocab_size += 1
            print(f"Vocab size: {current_vocab_size}, merged pair: {best_pair}, freq: {stats[best_pair]}")

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
        for a, b in self.merges:
            i = 0
            while i < len(token_list) - 1:
                if token_list[i] == a and token_list[i+1] == b:
                    token_list[i] = a + b
                    del token_list[i+1]
                else:
                    i += 1
        return token_list

    def encode(self, text):
        unk_id = self.get_unk_id()
        token_ids = []

        for segment in self._segments(text):
            if segment in self._special_token_set:
                token_ids.append(self.vocab.get(segment, unk_id))
                continue
            words = self._split_words(segment)
            for w in words:
                chars = list(w)
                tokens = self.apply_merges(chars)
                for token in tokens:
                    token_ids.append(self.vocab.get(token, unk_id))

        return token_ids

    def decode(self, token_ids):
        tokens = []
        for tid in token_ids:
            token = self.inverse_vocab.get(tid, "<|UNK|>")
            tokens.append(token)

        text = "".join(tokens)
        return text

    def save(self, dirpath):
        data = {
            "target_vocab_size": self.target_vocab_size,
            "special_tokens": self.special_tokens,
            "merges": self.merges,
            "vocab": self.vocab,
        }
        os.makedirs(dirpath, exist_ok=True)
        filepath = os.path.join(dirpath, "tokenizer.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load(self, dirpath):
        filepath = os.path.join(dirpath, "tokenizer.json")
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.target_vocab_size = data["target_vocab_size"]
        self.special_tokens = data["special_tokens"]
        self.merges = [tuple(m) for m in data["merges"]]
        self.vocab = data["vocab"]
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}
        self._special_token_set = set(self.special_tokens)

    def get_vocab_size(self):
        return len(self.vocab)

    def get_unk_id(self):
        return self.vocab.get("<|UNK|>", 1)