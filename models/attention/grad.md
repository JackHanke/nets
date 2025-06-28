## Symbolically Calculating the gradients for Multi Head Attention

Multihead Attention is define as:

$$\text{Multihead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O$$
$$\text{where } \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$
$$\text{and } \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$


## Loss setup

As I don't intend to do any real training with this module, we just compute MSE between "ground truth" attention values $\hat{M}$. 

$$\cal{L} = \frac{1}{2}\|M - \hat{M} \|_2$$


## Goal

We need to compute the following partials for optimization:

$$\frac{\partial \cal{L}}{\partial W^O}, \frac{\partial \cal{L}}{\partial W^Q}, \frac{\partial \cal{L}}{\partial W^K}, \frac{\partial \cal{L}}{\partial W^V}$$

## Useful Identity

If $\cal{L}$ is a scalar valued function that involves summing over entries of some $n \times m$ matrix $A$, and $A=BC$, then

$$\frac{\partial \cal{L}}{\partial B} = \frac{\partial \cal{L}}{\partial A} C^T$$

and 

$$\frac{\partial \cal{L}}{\partial C} = B^T \frac{\partial \cal{L}}{\partial A}$$

>Definitions for $A$ and $B$ only apply to this section. These letters will be used differently in another section.

## Intermediate Values

We define the following intermediate values to simplify gradient calculations.

$$M = \text{Multihead}(Q,K,V) = CW^O$$

$$head_i = S_i \bar{V}_i$$

where

$$T_i = \bar{Q}_i \bar{K}_i^T$$

$$S_i = \text{softmax}\left(\frac{T_i}{\sqrt{d_k}}\right)$$

$$\bar{Q}_i = QW_i^Q $$

$$\bar{K}_i = KW_i^K$$

$$\bar{V}_i = VW_i^V$$


## Gradients Summary

Here we summarize of the gradients, resolving all partials and equalities.

For simplicity, define

$$A = \frac{\partial \cal{L}}{\partial C} = (M - \hat{M}) (W^O)^T$$

and

$$B_i = \frac{\partial S_i}{\partial T_i} = \frac{1}{\sqrt{d_k}}\text{softmax}'\left(\frac{T_i}{\sqrt{d_k}} \right)$$

This gives us the results.

$$\frac{\partial \cal{L}}{\partial W^O} = C^T (M - \hat{M})$$

$$\frac{\partial \cal{L}}{\partial W^V_i} = V^T S^T_i A [d_k (i-1): d_{k} i] $$

$$\frac{\partial \cal{L}}{\partial W^Q_i} = Q^T A [d_k (i-1): d_{k} i] \bar{V}^T_i B_i \bar{K}_i$$

$$\frac{\partial \cal{L}}{\partial W^K_i} = K^T \left(\bar{V}^T_i A [d_k (i-1): d_{k} i] \bar{V}^T_i B_i\right)^T $$


## Gradients Calculation

This section details the step-by-step backprop variable chaining. I organize the identities into computation layers defined by the intermediate value identities above..

First layer of backprop:

$$\frac{\partial \cal{L}}{\partial M} = (M - \hat{M})$$

Second layer of backprop:

$$\text{*} \frac{\partial \cal{L}}{\partial W^O} = C^T \frac{\partial \cal{L}}{\partial M} $$

$$\frac{\partial \cal{L}}{\partial C} = \frac{\partial \cal{L}}{\partial M} (W^O)^T $$

$$\frac{\partial \cal{L}}{\partial \text{head}_i} = \frac{\partial \cal{L}}{\partial C} [d_k (i-1): d_{k} i]$$

using Python slice notation.

Third layer of backprop:

$$\frac{\partial \cal{L}}{\partial S_i} =  \frac{\partial \cal{L}}{\partial \text{head}_i} \bar{V}^T_i$$

$$\frac{\partial \cal{L}}{\partial \bar{V}_i} = S^T_i \frac{\partial \cal{L}}{\partial \text{head}_i} $$

Fourth layer of backprop:

$$\text{*}\frac{\partial \cal{L}}{\partial W^V_i} = V^T \frac{\partial \cal{L}}{\partial \bar{V}_i} $$

$$\frac{\partial \cal{L}}{\partial T_i} = \frac{\partial \cal{L}}{\partial S_i}\frac{1}{\sqrt{d_k}}\text{softmax}'(\frac{1}{\sqrt{d_k}}T_i)$$

$$\text{where }\text{softmax}'(\frac{T_i}{\sqrt{d_k}}) = \text{softmax}(\frac{T_i}{\sqrt{d_k}}) (1-\text{softmax}(\frac{T_i}{\sqrt{d_k}}))$$

Fifth layer of backprop:

$$\frac{\partial \cal{L}}{\partial \bar{Q}_i} = \frac{\partial \cal{L}}{\partial T_i} \bar{K}_i$$

$$\frac{\partial \cal{L}}{\partial \bar{K}^T_i} = \bar{V}^T_i \frac{\partial \cal{L}}{\partial T_i} $$

Sixth layer of backprop:

$$\text{*}\frac{\partial \cal{L}}{\partial W^Q_i} = Q^T \frac{\partial \cal{L}}{\partial \bar{Q}_i} $$

$$\text{*}\frac{\partial \cal{L}}{\partial W^K_i} = K^T \left(\frac{\partial \cal{L}}{\partial \bar{K}^T_i}\right)^T $$


