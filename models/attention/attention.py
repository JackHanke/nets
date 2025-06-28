## NOTE this is an academic excersize, and is not expected to be memory efficient (or used at all)
# formal variable definitions and gradient calculations in ./models/attention/grad.md

import numpy as np
np.random.seed(1)

# Multi Head Attention implementation
class MultiHeadAttention:
    def __init__(self, 
            d_model:int, 
            h: int, 
            softmax: callable,
            w_q: list[np.array] = None,
            w_k: list[np.array] = None,
            w_v: list[np.array] = None,
            w_o: np.array = None,
        ):

        self.d_model = d_model
        # number of heads h
        self.h = h
        self.softmax = softmax
        # head dims work out check
        assert (d_model % h) == 0
        self.d_k = d_model // h

        # initialize weight matrices for all heads
        if w_q is None:
            self.query_weights_list = [np.random.normal(loc=0, scale=1, size=(self.d_model, self.d_k)) for i in range(h)]
        elif w_q is not None:
            self.query_weights_list = w_q

        if w_k is None:
            self.key_weights_list = [np.random.normal(loc=0, scale=1, size=(self.d_model, self.d_k)) for i in range(h)]
        elif w_k is not None:
            self.key_weights_list = w_k

        if w_v is None:
            self.value_weights_list = [np.random.normal(loc=0, scale=1, size=(self.d_model, self.d_k)) for i in range(h)]
        elif w_v is not None:
            self.value_weights_list = w_v

        # final output weights
        if w_o is None:
            self.output_weights = np.random.normal(loc=0, scale=1, size=(self.d_model, self.d_model))
        elif w_o is not None:
            self.output_weights = w_o
    
    def _scaled_dot_attention(
            self,
            activation_q: np.array,
            activation_k: np.array,
            activation_v: np.array,
            q_bar_list: list,
            k_bar_list: list,
            v_bar_list: list,
            t_list: list,
            s_list: list,
            i: int,
            include: bool,
        ):
        # bar functions
        q_bar = np.matmul(activation_q, self.query_weights_list[i])
        # print(f'w_q ({self.query_weights_list[i].shape}) :\n{self.query_weights_list[i]}')
        k_bar = np.matmul(activation_k, self.key_weights_list[i])
        v_bar = np.matmul(activation_v, self.value_weights_list[i])
        if include:
            q_bar_list.append(q_bar)
            k_bar_list.append(k_bar)
            v_bar_list.append(v_bar)

        # intermediate t value
        k_t_bar = np.transpose(k_bar, axes=(0,2,1))
        t = np.matmul(q_bar, k_t_bar)
        if include:
            t_list.append(t)
        
        # softmaxed matrix s
        s = self.softmax.function(x=t/np.sqrt(self.d_k))

        if include:
            s_list.append(s)

        # attention value head
        head = np.matmul(s, v_bar)
        return head

    # forward pass for multihead attention module
    def _forward(self, 
            activation_q: np.array, 
            activation_k: np.array, 
            activation_v: np.array, 
            include=False,
        ):

        head_list = []
        if include:
            # intermediate values define in ./models/attention/grad.md
            q_bar_list = []
            k_bar_list = []
            v_bar_list = []
            t_list = []
            s_list = []
        else:
            q_bar_list = None
            k_bar_list = None
            v_bar_list = None
            t_list = None
            s_list = None

        # not vectorized bc too bad
        for i in range(self.h):
            # scaled dot product attention for specific head
            head = self._scaled_dot_attention(
                activation_q = activation_q,
                activation_k = activation_k,
                activation_v = activation_v,
                q_bar_list = q_bar_list,
                k_bar_list = k_bar_list,
                v_bar_list = v_bar_list,
                t_list = t_list,
                s_list = s_list,
                i = i,
                include = include,
            )
            head_list.append(head)

        # temp value C
        C = np.concatenate(head_list, axis=2)
        # print(f'c shape: {C.shape}')
        multi_head_attention_vals = np.matmul(C, self.output_weights)

        if include:
            return multi_head_attention_vals, q_bar_list, k_bar_list, v_bar_list, t_list, s_list, C
        return multi_head_attention_vals


    # backward pass for multi head attention module
    def _backward(self, 
            activation_q: np.array, 
            activation_k: np.array, 
            activation_v: np.array, 
            label: np.array,
        ):
        
        # forward pass for activations
        multi_head_attention_vals, q_bar_list, k_bar_list, v_bar_list, t_list, s_list, C = self._forward(
            activation_q = activation_q, 
            activation_k = activation_k, 
            activation_v = activation_v, 
            include=True
        )
        
        ## compute gradients 
        grad_output_weights = np.matmul(np.transpose(C, axes=(0,2,1)), (multi_head_attention_vals-label))

        mean_grad_weights_k_list = []
        mean_grad_weights_v_list = []
        mean_grad_weights_q_list = []

        # define intermediate matrix A 
        A = np.matmul((multi_head_attention_vals-label), np.transpose(self.output_weights))

        # for each head i
        for i in range(self.h):
            # define intermediate matrix B and slice of A
            B = self.softmax.function_prime(t_list[i]/np.sqrt(self.d_k))/np.sqrt(self.d_k)
            A_slice = A[:,:,self.d_k*i:self.d_k*(i+1)]

            ## NOTE had to do this bc only two args at a time

            # compute gradients for v
            grad_weights_v = np.transpose(activation_v, axes=(0,2,1))
            for term in [
                    np.transpose(s_list[i], axes=(0,2,1)),
                    A_slice,
                ]:
                grad_weights_v = np.matmul(grad_weights_v, term)
                
            # compute gradients for q
            grad_weights_q = np.transpose(activation_q, axes=(0,2,1))
            for term in [
                    A_slice,
                    np.transpose(v_bar_list[i], axes=(0,2,1)),
                    B,
                    k_bar_list[i],
                ]:
                # print(f'term shape: {term.shape}')
                grad_weights_q = np.matmul(grad_weights_q, term)

            # compute gradients for k
            temp_prod = np.transpose(v_bar_list[i], axes=(0,2,1))
            for term in [
                    A_slice,
                    np.transpose(v_bar_list[i], axes=(0,2,1)),
                    B,
                ]:
                temp_prod = np.matmul(temp_prod, term)
            grad_weights_k = np.matmul(np.transpose(activation_k, axes=(0,2,1)), np.transpose(temp_prod, axes=(0,2,1)))

            # dimension checks
            assert grad_weights_v.shape == (activation_q.shape[0], self.d_model, self.d_k)
            assert grad_weights_q.shape == (activation_q.shape[0], self.d_model, self.d_k)
            assert grad_weights_k.shape == (activation_q.shape[0], self.d_model, self.d_k)
            
            # mean grads over batch
            mean_grad_weights_v = np.mean(grad_weights_v, axis=0)
            mean_grad_weights_q = np.mean(grad_weights_q, axis=0)
            mean_grad_weights_k = np.mean(grad_weights_k, axis=0)

            # append to grads list
            mean_grad_weights_v_list.append(mean_grad_weights_v)
            mean_grad_weights_q_list.append(mean_grad_weights_q)
            mean_grad_weights_k_list.append(mean_grad_weights_k)

        # mean output weights grads 
        mean_grad_output_weights = np.mean(grad_output_weights, axis=0)
        assert mean_grad_output_weights.shape == (self.d_model, self.d_model)

        return mean_grad_output_weights, mean_grad_weights_q_list, mean_grad_weights_k_list, mean_grad_weights_v_list

