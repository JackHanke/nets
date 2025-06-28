import torch 
import torch.nn as nn 
import numpy as np
import math

# test custom attention mechanism forward pass against pytorch implementation
def attention_forward_pass_test(
        data: np.array,
        label: np.array,
        embedding_dim: int,
        num_heads: int,
        seed: int = 1,
    ):
    torch.manual_seed(seed)

    # tensorize data
    tensorized_data = torch.Tensor(data)
    tensorized_label = torch.Tensor(label)
    
    # run multi head attention on data
    torch_attention = nn.MultiheadAttention(
        embedding_dim, 
        num_heads, 
        batch_first=True,
        bias=False,
    )
    torch_result, _ = torch_attention(tensorized_data, tensorized_data, tensorized_data, need_weights=False)

    # get initialized weights 
    in_weights = torch_attention.in_proj_weight
    w_q, w_k, w_v = in_weights.chunk(3)
    w_o = torch_attention.out_proj.weight

    return torch_result, w_q, w_k, w_v, w_o

# test custom attention mechanism backward pass against pytorch implementation
def attention_update_test(
        data: np.array,
        label: np.array,
        embedding_dim: int,
        num_heads: int,
        lr: float,
        seed: int = 1,
    ):

    torch.manual_seed(seed)

    # tensorize data
    tensorized_data = torch.Tensor(data)
    tensorized_label = torch.Tensor(label)
    # run multi head attention on data
    torch_attention = nn.MultiheadAttention(
        embedding_dim, 
        num_heads, 
        batch_first=True,
        bias=False,
    )
    # just mse loss and sgd
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(torch_attention.parameters(), lr=lr)

    torch_result, _ = torch_attention(tensorized_data, tensorized_data, tensorized_data, need_weights=False)

    # compute loss
    loss = loss_fn(torch_result, tensorized_label)
    print(f'True loss: {loss.item()}')
    # compute gradients and update
    loss.backward()
    optimizer.step()

    # get updated weights 
    in_weights = torch_attention.in_proj_weight
    w_q, w_k, w_v = in_weights.chunk(3)
    w_o = torch_attention.out_proj.weight
    
    return torch_result, w_q, w_k, w_v, w_o
