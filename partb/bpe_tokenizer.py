import json
import re
import os
import time
from collections import defaultdict

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=None):
        self.target_vocab_size = vocab_size
        self.special_tokens = special_tokens or ["<|PAD|>", "<|UNK|>", "<|SOS|>", "<|EOS|>"]
        self.merges = []
        self.vocab = {}
        self.inverse_vocab = {}
        self._special_token_set = set()
        self._segment_boundary_pattern = None
        self._update_segment_pattern()

    def _update_segment_pattern(self):
        self._special_token_set = set(self.special_tokens)
        if self.special_tokens:
            escaped = sorted((re.escape(token) for token in self.special_tokens), key=len, reverse=True)
            pattern = "|".join(escaped)
            self._segment_boundary_pattern = re.compile(pattern)
        else:
            self._segment_boundary_pattern = None

    def _segments(self, text):
        # Split only around special tokens. Keep normal text chunks intact (including spaces)
        if not self._segment_boundary_pattern:
            return [text] if text else []
        segments = []
        last_index = 0

        for match in self._segment_boundary_pattern.finditer(text):
            start, end = match.span()
            if start > last_index:
                segments.append(text[last_index:start])
            segments.append(match.group(0))
            last_index = end

        if last_index < len(text):
            segments.append(text[last_index:])

        return [segment for segment in segments if segment]

    def get_stats(self, splits):
        counts = defaultdict(int)
        for symbols, freq in splits.items():
            prev = symbols[0]
            for sym in symbols[1:]:
                counts[(prev, sym)] += freq
                prev = sym
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

    def train(self, corpus, max_time_seconds=10000):
        segment_freqs = defaultdict(int)
        if isinstance(corpus, list):
            lines = corpus
        else:
            lines = corpus.splitlines()

        for line in lines:
            for segment in self._segments(line):
                if segment in self._special_token_set:
                    continue
                segment_freqs[segment] += 1

        splits = defaultdict(int)
        for segment, freq in segment_freqs.items():
            splits[tuple(segment)] += freq

        char_vocab = set()
        for symbols in splits:
            for symbol in symbols:
                char_vocab.add(symbol)

        for cp in range(0x0900, 0x0980): # Add devnagiri characters to the character vocabulary
            char_vocab.add(chr(cp))

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
            # print(f"Current vocab size: {current_vocab_size}, Merged pair: {best_pair}")

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
        unk_id = self.vocab.get("<|UNK|>", 1)
        token_ids = []

        for segment in self._segments(text):
            if segment in self._special_token_set:
                token_ids.append(self.vocab.get(segment, unk_id))
                continue
            chars = list(segment)
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
        self._update_segment_pattern()

    def get_vocab_size(self):
        return len(self.vocab)

    def get_unk_id(self):
        return self.vocab.get("<|UNK|>", -1)