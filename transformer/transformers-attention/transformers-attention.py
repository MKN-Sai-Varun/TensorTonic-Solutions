import torch
import torch.nn.functional as F
import math


def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    d_k=Q.size(-1)

    score=torch.matmul(Q,K.transpose(-2,-1))
    score=score/math.sqrt(d_k)
    attention_weight=F.softmax(score,dim=-1)
    output=torch.matmul(attention_weight,V)
    return output
    