import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List

def scaled_dot_product_attention(query,key,value,mask=None,mode='standard',S=1):
    "Size of mask = (B,L)"
    B,H,L,D = query.shape
    x = torch.matmul(query,key.transpose(-2,-1)) / D**0.5
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(2)  # required size (B, 1, 1, L)
        x = x + mask
    if mode == 'tanh-clipped':
        x = torch.tanh(x) * S

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
        self.W_q = nn.Linear(d_model,d_model)
        self.W_k = nn.Linear(d_model,d_model)
        self.W_v = nn.Linear(d_model,d_model)
        self.out_proj = nn.Linear(d_model,d_model)


    def split_heads(self,x):
        B,L,_ = x.shape
        x = x.view(B,L,self.num_heads,self.d_k)
        return x.transpose(1,2)
    
    def merge_heads(self,x):
        B,H,L,D_k = x.shape
        x = x.transpose(1,2).contiguous().view(B,L,self.d_model)
        return x
    
    def forward(self, x, mask=None, mode='standard', S=1):
        B,L,_ = x.shape
        Q = self.split_heads(self.W_q(x))
        K = self.split_heads(self.W_k(x))
        V = self.split_heads(self.W_v(x))
        attn_output, attn_wts = scaled_dot_product_attention(Q,K,V,mask,mode,S)
        attn_output = self.merge_heads(attn_output)
        output = self.out_proj(attn_output)
        return output, attn_wts

class FFN(nn.Module):
    def __init__(self,d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,4*d_model),
            nn.GELU(),
            nn.Linear(4*d_model,d_model)
        )

    def forward(self,x):
        return self.net(x)
    
class TransformerBlock(nn.Module):
    def __init__(self,d_model,num_heads):
        super().__init__()
        self.mha = MultiHeadAttention(d_model,num_heads)
        self.ffn = FFN(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self,x,mask=None,mode='standard',S=1):
        x = self.mha(self.norm1(x),mask,mode,S) + x
        x = self.ffn(self.norm2(x)) + x
        return x
    














class LanguageModel(nn.Module):
    """
    This is a stub class for the assignment.
    Feel free to change the function signatures (including that of __init__, forward) as you need them.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Build the LanguageModel based on the config.
        """
        self.config = config
        super().__init__()

    def set_weights(self, weights: Dict[str, Any]):
        """
        Set the model's weights based on the provided dictionary.
        The weights dictionary will contain all necessary parameters to initialize the model's layers.
        You should ensure that the weights are correctly assigned to the corresponding layers in your model.

        Parameters:
            - weights: A dictionary containing the model's weights. The structure of this dictionary will depend on how you design your model.
        """
        raise NotImplementedError("Implement set_weights as described in assignment document")

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
        raise NotImplementedError("Implement forward as described in assignment document")



def load_model(config: Dict[str, Any], weights: Dict[str, Any]):
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function inputs config and weights and outputs a nn.Module derived object.
    """

    model = LanguageModel(config)
    model.set_weights(weights)

    return model


def collate_fn(batch: Dict[str, List[torch.tensor]]) -> Dict[str, torch.Tensor]:
    """
    This is a sample code. Replace with your own.
    However, DO NOT CHANGE THE SIGNATURE OF THIS FUNCTION.
    Ensure that the function takes in a batch of data and outputs a dictionary of tensors ready to be fed into the model.
    """
    PAD_ID = 0  # Assume 0 is the padding token ID
    raise NotImplementedError("Implement collate_fn as described in assignment document")
