class BPETokenizer:
    def __init__(self, vocab_size, special_tokens=None):
        raise NotImplementedError("BPETokenizer initialization not implemented yet.")

    def train(self, corpus):
        raise NotImplementedError("Training method not implemented yet.")
    
    def encode(self, text):
        raise NotImplementedError("Encoding method not implemented yet.")

    def decode(self, token_ids):
        raise NotImplementedError("Decoding method not implemented yet.")

    def save(self, filepath):
        raise NotImplementedError("Save method not implemented yet.")

    def load(self, filepath):
        raise NotImplementedError("Load method not implemented yet.")
    
    def get_vocab_size(self):
        raise NotImplementedError("Get vocab size method not implemented yet.")
    
    def get_unk_id(self):
        raise NotImplementedError("Get unk id method not implemented yet.")
