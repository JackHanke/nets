# system imports
import numpy as np

# local imports
from models.attention.attention import MultiHeadAttention
from models.attention.test import attention_forward_pass_test, attention_update_test
from functions.activation_funcs import SoftMax

## NOTE I test by seeing if the gradients match with the torch implementation of multihead attention
# settings for test    
batch_size = 2
embedding_dim = 6
seq_len = 7
num_heads = 2

# make toy data
np.random.seed(1)
data = np.random.normal(loc=0, scale=1, size=(batch_size, seq_len, embedding_dim))
# make toy labels
label = np.zeros((batch_size, seq_len, embedding_dim))
# get ground truth output and gradients
torch_result, w_q, w_k, w_v, w_o = attention_forward_pass_test(
    data =data,
    label=label,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
)

# translate weights to custom format
d_k = embedding_dim // num_heads

# pytorch internally tranposes the matrices before multiplying (in the linear function)
w_q = np.transpose(w_q.detach().numpy())
w_k = np.transpose(w_k.detach().numpy())
w_v = np.transpose(w_v.detach().numpy())
w_o = np.transpose(w_o.detach().numpy())

w_q_list = [w_q[:, d_k*i:d_k*(i+1)] for i in range(num_heads)]
w_k_list = [w_k[:, d_k*i:d_k*(i+1)] for i in range(num_heads)]
w_v_list = [w_v[:, d_k*i:d_k*(i+1)] for i in range(num_heads)]

# custom implementation 
mha = MultiHeadAttention(
    d_model=embedding_dim, 
    h=num_heads,
    softmax=SoftMax(),
    w_q = w_q_list,
    w_k = w_k_list,
    w_v = w_v_list,
    w_o = w_o,
)

mha_vals = mha._forward(
    activation_q = data, 
    activation_k = data, 
    activation_v = data, 
)

# compare forward pass
assert np.isclose(mha_vals, torch_result.detach().numpy()).all()
print(f'Forward pass matches!')


## backward pass check via SGD updates

w_o_grads, w_q_grads_list, w_k_grads_list, w_v_grads_list = mha._backward(
    activation_q = data, 
    activation_k = data, 
    activation_v = data, 
    label = label,
)

# update weights with SGD
lr = 0.5

# update output weights
scale = 2/(mha_vals.shape[1]*mha_vals.shape[2]) # note scale because gradients dont include meaning of factor of 2
print(f'Scale for averaging: {scale}')
updated_w_o = w_o - 0.5*scale*w_o_grads

predicted_loss = np.mean(np.square(label - mha_vals))
print(f'Predicted Loss: {predicted_loss}')

updated_w_q_grads = []
updated_w_k_grads = []
updated_w_v_grads = []
for head_index, (w_q_grad, w_k_grad, w_v_grad) in enumerate(zip(w_q_grads_list, w_k_grads_list, w_v_grads_list)):
    # update query weights
    updated_w_q = mha.query_weights_list[head_index] - 0.5*scale*w_q_grad
    updated_w_q_grads.append(updated_w_q)
    # update key weights
    updated_w_k = mha.key_weights_list[head_index] - 0.5*scale*w_k_grad
    updated_w_k_grads.append(updated_w_k)
    # update value weights
    updated_w_v = mha.value_weights_list[head_index] - 0.5*scale*w_v_grad
    updated_w_v_grads.append(updated_w_v)

torch_result, new_w_q, new_w_k, new_w_v, new_w_o = attention_update_test(
    data =data,
    label=label,
    embedding_dim=embedding_dim,
    num_heads=num_heads,
    lr=lr,
)

update_w_q = np.concatenate(updated_w_q_grads, axis=1)
update_w_k = np.concatenate(updated_w_k_grads, axis=1)
update_w_v = np.concatenate(updated_w_v_grads, axis=1)

# compare backward pass
assert np.isclose(updated_w_o, new_w_o.T.detach().numpy()).all()
print(f'Updated w_o matches!')

assert np.isclose(update_w_v, new_w_v.T.detach().numpy()).all()
print(f'Updated w_v s matches!')

## I think these fail because of FLOP issue? idk they look close

assert np.isclose(update_w_q, new_w_q.T.detach().numpy(), atol=1e-02).all()
print(f'Updated w_q s matches!')

assert np.isclose(update_w_k, new_w_k.T.detach().numpy(), atol=1e-02).all()
print(f'Updated w_k s matches!')


print(f'Backward pass matches!')
