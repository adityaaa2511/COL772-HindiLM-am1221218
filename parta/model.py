import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List

def scaled_dot_product_attention(query,key,value,mask=None,mode='standard',tau=1):
    "Size of mask = (B,L,L)"
    B,H,L,D = query.shape
    x = torch.matmul(query,key.transpose(-2,-1)) / D**0.5
    if mode == 'tanh-clipped':
        x = torch.tanh(x) * tau
    if mask is not None:
        mask = mask.unsqueeze(1)  # required size (B, 1, L, L) to broadcast across heads
        x = x + mask

    attn_wts = F.softmax(x,dim=-1)
    output = torch.matmul(attn_wts,value)
    return output, attn_wts

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model,d_model, bias=False)
        self.W_k = nn.Linear(d_model,d_model, bias=False)
        self.W_v = nn.Linear(d_model,d_model, bias=False)
        self.out_proj = nn.Linear(d_model,d_model, bias=False)
        # self.dropout = nn.Dropout(0.1)  

    def split_heads(self,x):
        B,L,_ = x.shape
        x = x.view(B,L,self.num_heads,self.d_k)
        return x.transpose(1,2)
    
    def merge_heads(self,x):
        B,H,L,D_k = x.shape
        x = x.transpose(1,2).contiguous().view(B,L,self.d_model)
        return x
    
    def forward(self, x, mask=None, mode='standard', tau=1):
        B,L,_ = x.shape
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        attn_output, attn_wts = scaled_dot_product_attention(Q,K,V,mask,mode,tau)
        attn_output = self.merge_heads(attn_output)
        output = self.out_proj(attn_output)
        # output = self.dropout(output)
        return output, attn_wts

class FFN(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            # nn.Dropout(0.1),
            nn.Linear(4*d_model,d_model),
            # nn.Dropout(0.1)
        )

    def forward(self,x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ffn = FFN(d_model)
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=True)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=True)

    def forward(self,x,mask=None,mode='standard',tau=1):
        attn_output, _ = self.mha(self.norm1(x),mask,mode,tau)
        x = attn_output + x
        x = self.ffn(self.norm2(x)) + x
        return x
    

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        B,L,D = x.shape
        return x + self.pe[:L, :].unsqueeze(0)


class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        super().__init__()
        self.config = config
        self.d_model = config['d_model']
        self.n_heads = config['n_heads']
        self.d_heads = config['d_head']
        self.n_layers = config['n_layers']
        self.vocab_size = config['vocab_size']
        self.mode = config.get('mode', 'standard')
        self.tau = config.get('tau', 1)

        self.W_vocab = nn.Embedding(self.vocab_size, self.d_model)
        self.layers = nn.ModuleList([TransformerBlock(self.d_model, self.n_heads) for _ in range(self.n_layers)])
        self.pos_encoding = SinusoidalPositionalEncoding(self.d_model)
        self.W_devocab = nn.Linear(self.d_model, self.vocab_size, bias=False)
        self.final_ln = nn.LayerNorm(self.d_model, elementwise_affine=True)
        # self.W_devocab.weight = self.W_vocab.weight

    def create_causal_mask(self, L, device):

        mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, -1e9)
        return mask

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        with torch.no_grad():

            self.W_vocab.weight.copy_(weights['W_vocab'].T)
            self.W_devocab.weight.copy_(weights['W_devocab'].T)
            self.final_ln.weight.copy_(weights['gamma_final'])
            self.final_ln.bias.copy_(weights['beta_final'])

            for i in range(self.n_layers):

                layer_id = i+1
                layer = self.layers[i]

                Wq_list = []
                Wk_list = []
                Wv_list = []

                for h in range(self.n_heads):

                    head_id = h+1

                    Wq = weights[f'W_{layer_id}_Q_{head_id}']
                    Wk = weights[f'W_{layer_id}_K_{head_id}']
                    Wv = weights[f'W_{layer_id}_V_{head_id}']

                    Wq_list.append(Wq)
                    Wk_list.append(Wk)
                    Wv_list.append(Wv)

                Wq_cat = torch.cat(Wq_list, dim=0)
                Wk_cat = torch.cat(Wk_list, dim=0)
                Wv_cat = torch.cat(Wv_list, dim=0)

                layer.mha.W_q.weight.copy_(Wq_cat.T)
                layer.mha.W_k.weight.copy_(Wk_cat.T)
                layer.mha.W_v.weight.copy_(Wv_cat.T)

                Wo = weights[f'W_{layer_id}_O']
                layer.mha.out_proj.weight.copy_(Wo.T)

                layer.ffn.net[0].weight.copy_(weights[f'W_{layer_id}_up'].T)
                layer.ffn.net[0].bias.copy_(weights[f'b_{layer_id}_up'])
                layer.ffn.net[2].weight.copy_(weights[f'W_{layer_id}_down'].T)
                layer.ffn.net[2].bias.copy_(weights[f'b_{layer_id}_down'])

                layer.norm1.weight.copy_(weights[f'gamma_{layer_id}_1'])
                layer.norm1.bias.copy_(weights[f'beta_{layer_id}_1'])
                layer.norm2.weight.copy_(weights[f'gamma_{layer_id}_2'])
                layer.norm2.bias.copy_(weights[f'beta_{layer_id}_2'])


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Implement the forward pass of the model. The output should be a tensor of shape (T, |Vocab|).

        Parameters:
            - input_ids: A tensor of shape (batch_size, sequence_len) containing token IDs.
            - attention_mask: A tensor of shape (batch_size, sequence_len) containing 1s for valid tokens and 0s for padding.

        Returns:
            - A tensor of shape (batch_size, sequence_len, vocab_size) containing the logits for each token in the vocabulary.
            Logits are the raw, unnormalized scores output by the model, which can be converted to probabilities using a softmax function.
        """

        B, L = input_ids.shape
        device = input_ids.device
        X = self.W_vocab(input_ids)
        X = self.pos_encoding(X)

        causal_mask = self.create_causal_mask(L, device) # Shape = (L, L)

        padding_mask = torch.zeros_like(attention_mask, dtype=torch.float, device=device) # Shape = (B, L)
        padding_mask = padding_mask.masked_fill(attention_mask == 0, -1e9)

        combined_mask = padding_mask.unsqueeze(1) + causal_mask # Shape = (B, L, L)

        for layer in self.layers:
            X = layer(X, combined_mask, self.mode, self.tau)

        X = self.final_ln(X)
        logits = self.W_devocab(X)
        return logits


def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)

    return model

def collate_fn(batch):
    PAD_ID = 0

    if isinstance(batch, dict) and "input_ids" in batch:
        input_ids = batch["input_ids"]
        attention_masks = batch["attention_mask"]
        labels = batch.get("labels", None)
        char_lens = batch.get("char_len", None)

    else:
        input_ids = [item["input_ids"] for item in batch]
        attention_masks = [item["attention_mask"] for item in batch]
        labels = [item.get("labels") for item in batch]
        char_lens = [item.get("char_len") for item in batch]

    max_len = max(len(ids) for ids in input_ids)

    padded_ids = []
    padded_masks = []
    padded_labels = []

    for i in range(len(input_ids)):
        ids = input_ids[i]
        mask = attention_masks[i]
        lbl = labels[i] if labels is not None else None

        pad_len = max_len - len(ids)

        padded_ids.append(torch.cat([ids, torch.full((pad_len,), PAD_ID)]))
        padded_masks.append(torch.cat([mask, torch.zeros(pad_len)]))

        if lbl is not None:
            padded_labels.append(torch.cat([lbl, torch.full((pad_len,), -100)]))

    result = {
        "input_ids": torch.stack(padded_ids).long(),
        "attention_mask": torch.stack(padded_masks).long(),
    }

    if labels is not None:
        result["labels"] = torch.stack(padded_labels).long()

    if char_lens is not None:
        result["char_len"] = char_lens

    return result
