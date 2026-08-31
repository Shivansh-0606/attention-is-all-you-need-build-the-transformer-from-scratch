"""
Attention Is All You Need: Build the Transformer From Scratch

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - build_token_to_id_vocab
def build_token_to_id_vocab(sentences, specials=('<pad>', '<bos>', '<eos>', '<unk>')):
    # TODO: build a token-to-id dict with specials first, then corpus tokens in first-seen order.
    
    token_to_id_vocab = {}

    for i in range(len(specials)):
        token_to_id_vocab[specials[i]] = i
    
    counter = len(specials)
    
    for word in sentences:
        for token in word.split():
            if token not in token_to_id_vocab:
                token_to_id_vocab[token] = counter
                counter+=1
            else:
                continue
    
    return token_to_id_vocab

# Step 2 - build_id_to_token_vocab
def build_id_to_token_vocab(token_to_id):
    # TODO: build the inverse id-to-token dictionary from token_to_id
    id_to_token_vocab = {}

    for key,value in token_to_id.items():
        id_to_token_vocab[value] = key
    
    return id_to_token_vocab

# Step 3 - encode_sentence_to_ids
def encode_sentence_to_ids(sentence, token_to_id, unk_token='<unk>'):
    # TODO: convert whitespace tokens of `sentence` to ids via `token_to_id`, using `unk_token`'s id for OOV
    
    tokens = sentence.split()

    arr = []

    for token in tokens:
        if token in token_to_id:
            arr.append(token_to_id[token])
        else:
            arr.append(token_to_id[unk_token])
    
    return arr

# Step 4 - decode_ids_to_tokens
def decode_ids_to_tokens(ids, id_to_token):
    # TODO: map each id in ids to its token string via id_to_token and return the list
    
    arr = []

    for i in ids:
        arr.append(id_to_token[i])
    
    return arr

# Step 5 - pad_id_sequence
def pad_id_sequence(ids, max_len, pad_id):
    # TODO: return a list of length exactly max_len, padding with pad_id or truncating.
    n = len(ids)

    if n < max_len:
        temp = n
        while temp!=max_len:
            ids.append(pad_id)
            temp+=1
    
        return ids

    elif n > max_len:
        temp = n
        
        while temp!=max_len:
            ids.pop()
            temp-=1
        
        return ids
    
    else:
        return ids

# Step 6 - stack_padded_sequences_to_batch
import torch

def stack_padded_sequences_to_batch(padded_sequences):
    """Stack a list of equal-length padded id sequences into a 2D LongTensor batch."""
    # TODO: stack padded id sequences into a (B, L) torch.long tensor
    return torch.LongTensor(padded_sequences)

# Step 7 - scale_embeddings_by_sqrt_d_model
import math
import torch

def scale_embeddings_by_sqrt_d_model(embeddings, d_model):
    """Scale a token embedding tensor by sqrt(d_model)."""
    # TODO: rescale embeddings by sqrt(d_model) as in the original Transformer paper
    return embeddings * math.sqrt(d_model)

# Step 8 - compute_positional_div_term
import torch

def compute_positional_div_term(d_model):
    # TODO: return a 1D FloatTensor of length d_model // 2 holding the sinusoidal frequency divisors

    i = torch.arange(0,d_model,2,dtype = torch.float32)

    return 10000**(-i/d_model)

# Step 9 - build_position_index_column
import torch

def build_position_index_column(max_len):
    """Return a (max_len, 1) float tensor of [0, 1, ..., max_len-1]."""
    # TODO: build a column vector of position indices from 0 to max_len-1

    return torch.arange(0,max_len,dtype = torch.float32).reshape(max_len,1)

# Step 10 - fill_even_indices_with_sin
import torch
def fill_even_indices_with_sin(pe, position, div_term):
    """Fill even feature indices of pe with sin(position * div_term)."""
    # TODO: write sin(position * div_term) into the even-indexed columns of pe and return it
    
    pe[:,0::2] = torch.sin(position * div_term)

    return pe

# Step 11 - fill_odd_indices_with_cos
import torch

def fill_odd_indices_with_cos(pe, position, div_term):
    # TODO: fill the odd-indexed columns of pe with cos(position * div_term)
    pe[:,1::2] = torch.cos(position * div_term)

    return pe

# Step 12 - build_sinusoidal_positional_encoding
import torch

def build_sinusoidal_positional_encoding(max_len, d_model):
    """Assemble the (max_len, d_model) sinusoidal positional encoding matrix."""
    # TODO: build the (max_len, d_model) sinusoidal positional encoding matrix
    position_encoded_matrix = torch.zeros(max_len ,d_model ,dtype = torch.float32)

    div_term = compute_positional_div_term(d_model)
    position_col = build_position_index_column(max_len)

    position_encoded_matrix = fill_even_indices_with_sin(position_encoded_matrix, position_col, div_term)

    position_encoded_matrix = fill_odd_indices_with_cos(position_encoded_matrix, position_col, div_term)

    return position_encoded_matrix

# Step 13 - add_positional_encoding_to_embeddings
import torch

def add_positional_encoding_to_embeddings(embedded_batch, positional_encoding):
    # TODO: add the first L rows of positional_encoding to embedded_batch and return the sum.
    L = embedded_batch.shape[1]
    
    if positional_encoding.dim() == 2:
        pe_slice = positional_encoding[:L , :]
    else:
        pe_slice = positional_encoding[:,:L,:]
    
    return embedded_batch + pe_slice

# Step 14 - build_padding_mask
import torch

def build_padding_mask(token_ids, pad_id):
    """Return a (B, 1, 1, L) bool mask: True where token_ids != pad_id."""
    # TODO: build a boolean mask marking non-pad positions, shaped for broadcasting against attention scores
    mask = torch.where(token_ids!=pad_id,True,False)
    return mask.unsqueeze(1).unsqueeze(1)

# Step 15 - build_causal_mask
import torch

def build_causal_mask(seq_len):
    """Return a (1, 1, seq_len, seq_len) bool mask, True on and below diagonal."""
    # TODO: build a lower-triangular boolean causal mask of shape (1, 1, seq_len, seq_len)
    return torch.tril(torch.ones((1,1,seq_len,seq_len),dtype = torch.bool))

# Step 16 - combine_padding_and_causal_masks
import torch

def combine_padding_and_causal_masks(padding_mask, causal_mask):
    # TODO: combine a (B,1,1,L) padding mask with a (1,1,L,L) causal mask into (B,1,L,L).
    # & = Broadcasting and bitwise comaprison 
    return padding_mask & causal_mask

# Step 17 - compute_raw_attention_scores
import torch

def compute_raw_attention_scores(query, key):
    """Compute raw attention scores Q @ K^T over the last two dimensions."""
    # TODO: matmul query with the transpose of key over the last two axes
    return query @ key.transpose(-2,-1)

# Step 18 - scale_attention_scores
import torch
import math

def scale_attention_scores(scores, d_k):
    # TODO: divide raw attention scores by sqrt(d_k) to stabilize softmax inputs
    return scores / math.sqrt(d_k)

# Step 19 - mask_attention_scores_with_neg_inf
import torch

def mask_attention_scores_with_neg_inf(scores, mask):
    """Set entries of scores where mask is False to -inf."""
    # TODO: replace blocked positions of scores with negative infinity
    if mask is None:
        return scores
    return torch.where(mask , scores , float('-inf'))

# Step 20 - softmax_attention_weights
import torch
import torch.nn.functional as F

def softmax_attention_weights(masked_scores):
    # TODO: softmax over the last axis, zeroing rows that are entirely -inf
   return torch.nan_to_num(F.softmax(masked_scores,dim = -1),nan = 0.0)

# Step 21 - apply_attention_weights_to_values
import torch

def apply_attention_weights_to_values(attention_weights, value):
    """Multiply attention weights by the value matrix to produce context vectors."""
    # TODO: combine attention weights (..., Lq, Lk) with value (..., Lk, d_v)
    return attention_weights @ value

# Step 22 - scaled_dot_product_attention
import torch

def scaled_dot_product_attention(query, key, value, mask=None):
    """Run scaled dot-product attention; return (context, attention_weights)."""
    # TODO: chain raw scores, scale by sqrt(d_k), optionally mask, softmax, then mix values
    d_k = key.size(-1)

    scores = compute_raw_attention_scores(query, key)

    scaled_scores = scale_attention_scores(scores, d_k)

    masked_scores = mask_attention_scores_with_neg_inf(scaled_scores, mask)

    attention_weights =  softmax_attention_weights(masked_scores)

    context_vector = apply_attention_weights_to_values(attention_weights, value)

    return (context_vector,attention_weights)

# Step 23 - split_last_dim_into_heads
import torch

def split_last_dim_into_heads(tensor, num_heads):
    # TODO: reshape (B, L, d_model) into (B, L, num_heads, d_model // num_heads)
    B,L,d_model = tensor.shape
    

    return tensor.reshape(B,L,num_heads,d_model//num_heads)

# Step 24 - transpose_heads_before_sequence
import torch

def transpose_heads_before_sequence(split_tensor):
    # TODO: rearrange (B, L, num_heads, d_k) into (B, num_heads, L, d_k).
    return torch.transpose(split_tensor , 1 , 2)

# Step 25 - merge_heads_back_to_model_dim
import torch

def merge_heads_back_to_model_dim(multi_head_tensor):
    # TODO: merge the head axis back into the feature axis to reconstruct d_model
    multi_head_tensor = torch.transpose(multi_head_tensor,1,2)

    B,L,H,d_k = multi_head_tensor.shape
    
    return multi_head_tensor.reshape(B,L,H*d_k)

# Step 26 - apply_linear_projection
def apply_linear_projection(x, weight, bias):
    # TODO: return x @ weight^T + bias (bias may be None) with shape (..., out_features)
    out = x @ weight.T

    if bias is not None:
        out+=bias
    return out

# Step 27 - project_to_query_key_value
def project_to_query_key_value(x, w_q, b_q, w_k, b_k, w_v, b_v):
    # TODO: project x into separate query, key, and value tensors via three linear layers
    q = apply_linear_projection(x, w_q, b_q)

    k = apply_linear_projection(x, w_k, b_k)
    
    v = apply_linear_projection(x, w_v, b_v)

    return (q,k,v)

# Step 28 - split_qkv_into_heads
import torch

def split_qkv_into_heads(q, k, v, num_heads):
    # TODO: split each of q, k, v into (B, num_heads, L, d_k) and return as a tuple
    split_q =  split_last_dim_into_heads(q, num_heads)
    q_h = transpose_heads_before_sequence(split_q)

    split_k = split_last_dim_into_heads(k, num_heads)
    k_h = transpose_heads_before_sequence(split_k)

    split_v = split_last_dim_into_heads(v, num_heads)
    v_h = transpose_heads_before_sequence(split_v)

    return (q_h , k_h , v_h)

# Step 29 - multi_head_scaled_dot_product_attention
import torch

def multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask=None):
    # TODO: run scaled dot-product attention over per-head Q, K, V and return (context, weights)
    return scaled_dot_product_attention(q_h, k_h, v_h, mask)

# Step 30 - merge_heads_and_project_output
import torch

def merge_heads_and_project_output(context, w_o, b_o):
    # TODO: merge the head axis back into d_model and apply the output linear projection.
    merged_heads = merge_heads_back_to_model_dim(context)
    final_multi_head_attentions =  apply_linear_projection(merged_heads, w_o, b_o)

    return final_multi_head_attentions

# Step 31 - assemble_multi_head_attention_forward
def assemble_multi_head_attention_forward(query, key, value, w_q, w_k, w_v, w_o, num_heads, mask=None):
    # TODO: project Q/K/V, split into heads, run scaled dot-product attention, merge heads, output projection.
    
    q = apply_linear_projection(query, w_q, None)

    k = apply_linear_projection(key, w_k, None)

    v = apply_linear_projection(value, w_v, None)

    q_h , k_h , v_h = split_qkv_into_heads(q, k, v, num_heads)

    context_vector,attention_weights = multi_head_scaled_dot_product_attention(q_h, k_h, v_h, mask)

    attention_output = merge_heads_and_project_output(context_vector, w_o, None)

    return attention_output

# Step 32 - apply_ffn_first_linear_and_relu
import torch
import torch.nn.functional as F

def apply_ffn_first_linear_and_relu(x, w1, b1):
    # TODO: project x by w1, add b1, then apply a ReLU activation.
    out = x @ w1 + b1

    return F.relu(out)

# Step 33 - apply_ffn_second_linear
import torch

def apply_ffn_second_linear(hidden, w2, b2):
    # TODO: project hidden (..., d_ff) back to (..., d_model) via w2 and b2.
    return hidden @ w2 + b2

# Step 34 - position_wise_feed_forward_network
def position_wise_feed_forward_network(x, w1, b1, w2, b2):
    # TODO: compose the two FFN linears with a ReLU in between, returning shape (B, T, d_model).
    
    hidden = apply_ffn_first_linear_and_relu(x, w1, b1)  


    return apply_ffn_second_linear(hidden, w2, b2)

# Step 35 - compute_layer_norm_mean_and_variance
import torch

def compute_layer_norm_mean_and_variance(x):
    # TODO: return (mean, variance) reduced over the last dim with shape (..., 1)
    mean = torch.mean(x,dim = -1 , keepdim = True)
    variance = torch.var(x,dim = -1 , keepdim = True , unbiased = False)

    return (mean , variance)

# Step 36 - normalize_and_scale_with_gamma_beta
import torch

def normalize_and_scale_with_gamma_beta(x, gamma, beta, eps=1e-5):
    # TODO: standardize x along the last axis then apply gamma and beta affine transform
    mean , var =   compute_layer_norm_mean_and_variance(x)

    std_x = (x - mean) / torch.sqrt(var+eps)

    return gamma * std_x + beta

# Step 37 - apply_residual_add_and_norm
import torch

def apply_residual_add_and_norm(residual_input, sublayer_output, gamma, beta, eps=1e-5):
    # TODO: combine the residual with the sublayer output and layer-normalize the result.
    
    added_output = residual_input + sublayer_output
    
    return normalize_and_scale_with_gamma_beta(added_output, gamma, beta, eps=1e-5)

# Step 38 - apply_dropout_with_keep_mask
def apply_dropout_with_keep_mask(x, keep_mask, keep_prob):
    # TODO: multiply x by the boolean keep_mask and rescale by 1/keep_prob.
    
    return (x * keep_mask) / keep_prob

# Step 39 - encoder_layer_self_attention_sublayer
def encoder_layer_self_attention_sublayer(x, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head self-attention on x and wrap with residual add-and-norm.
    attn_out = assemble_multi_head_attention_forward(
        query = x,
        key = x,
        value = x, 
        w_q = w_q, 
        w_k = w_k, 
        w_v = w_v, 
        w_o = w_o, num_heads = num_heads,
        mask=src_mask
    )

    return apply_residual_add_and_norm(
        residual_input = x, 
        sublayer_output = attn_out, 
        gamma = gamma, 
        beta = beta, 
        eps=1e-5
    )

# Step 40 - encoder_layer_feed_forward_sublayer
def encoder_layer_feed_forward_sublayer(x, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on x and wrap it with residual add-and-norm.
    sublayer_output =  position_wise_feed_forward_network(x, w1, b1, w2, b2)

    return apply_residual_add_and_norm(x, sublayer_output, gamma, beta, eps=1e-5)

# Step 41 - assemble_encoder_layer
def assemble_encoder_layer(x, layer_params, num_heads, src_mask):
    # TODO: chain the self-attention sublayer and the feed-forward sublayer using layer_params.
    attn_output = encoder_layer_self_attention_sublayer(
    x, 
    layer_params['w_q'], 
    layer_params['w_k'], 
    layer_params['w_v'], 
    layer_params['w_o'], 
    layer_params['attn_gamma'],
    layer_params['attn_beta'], 
    num_heads, 
    src_mask
    )

    return encoder_layer_feed_forward_sublayer(
        attn_output, 
        layer_params['w1'], 
        layer_params['b1'], 
        layer_params['w2'], 
        layer_params['b2'], 
        layer_params['ffn_gamma'], 
        layer_params['ffn_beta']
        )

# Step 42 - stack_encoder_layers
def stack_encoder_layers(x, encoder_layer_params_list, num_heads, src_mask):
    # TODO: sequentially apply each encoder layer to the running hidden state and return the final tensor.
    for layer_params in encoder_layer_params_list:
       x = assemble_encoder_layer(x, layer_params, num_heads, src_mask)
    return x

# Step 43 - decoder_layer_masked_self_attention_sublayer
import torch

def decoder_layer_masked_self_attention_sublayer(y, w_q, w_k, w_v, w_o, gamma, beta, num_heads, tgt_mask):
    # TODO: run masked multi-head self-attention on y and wrap with residual add-and-norm.
    attn_out = assemble_multi_head_attention_forward(
    query = y, 
    key = y, 
    value = y, 
    w_q = w_q, 
    w_k = w_k, 
    w_v = w_v, 
    w_o = w_o, 
    num_heads = num_heads, 
    mask=tgt_mask
    )

    return apply_residual_add_and_norm(y, attn_out, gamma, beta, eps=1e-5)

# Step 44 - decoder_layer_cross_attention_sublayer
import torch

def decoder_layer_cross_attention_sublayer(y, encoder_output, w_q, w_k, w_v, w_o, gamma, beta, num_heads, src_mask):
    # TODO: run multi-head cross-attention (Q from y, K/V from encoder_output) and wrap with add-and-norm
    
    if src_mask is not None and src_mask.dim() == 2:
        mask = src_mask.unsqueeze(1).unsqueeze(1)
    else:
        mask = src_mask
    
    cross_attn_out = assemble_multi_head_attention_forward(
    query = y, 
    key = encoder_output, 
    value = encoder_output, 
    w_q = w_q, 
    w_k = w_k, 
    w_v = w_v, 
    w_o = w_o, 
    num_heads = num_heads, 
    mask=mask)
    
    return apply_residual_add_and_norm(y, cross_attn_out, gamma, beta, eps=1e-5)

# Step 45 - decoder_layer_feed_forward_sublayer
import torch

def decoder_layer_feed_forward_sublayer(y, w1, b1, w2, b2, gamma, beta):
    # TODO: run the position-wise FFN on y and wrap it with residual add-and-norm
    layer_output = position_wise_feed_forward_network(y, w1, b1, w2, b2)

    return apply_residual_add_and_norm(y, layer_output, gamma, beta, eps=1e-5)

# Step 46 - assemble_decoder_layer
def assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask):
    """Run a full decoder layer: masked self-attention, cross-attention, then FFN."""

    def get_param(candidates, default=None):
        for k in candidates:
            if k in layer_params:
                return layer_params[k]
        return default

    # --- 1. Masked Self-Attention Sublayer ---
    self_w_q = get_param(['self_attn_w_q', 'self_w_q', 'w_q_self', 'w_q1', 'w_q_1', 'w_q'])
    self_w_k = get_param(['self_attn_w_k', 'self_w_k', 'w_k_self', 'w_k1', 'w_k_1', 'w_k'])
    self_w_v = get_param(['self_attn_w_v', 'self_w_v', 'w_v_self', 'w_v1', 'w_v_1', 'w_v'])
    self_w_o = get_param(['self_attn_w_o', 'self_w_o', 'w_o_self', 'w_o1', 'w_o_1', 'w_o'])
    self_gamma = get_param(['self_attn_gamma', 'self_gamma', 'attn_gamma_1', 'gamma1', 'attn_gamma'])
    self_beta = get_param(['self_attn_beta', 'self_beta', 'attn_beta_1', 'beta1', 'attn_beta'])

    attn_output = decoder_layer_masked_self_attention_sublayer(
        y,
        self_w_q,
        self_w_k,
        self_w_v,
        self_w_o,
        self_gamma,
        self_beta,
        num_heads,
        tgt_mask
    )

    # --- 2. Cross-Attention Sublayer ---
    cross_w_q = get_param(['cross_attn_w_q', 'cross_w_q', 'w_q_cross', 'w_q2', 'w_q_2'])
    cross_w_k = get_param(['cross_attn_w_k', 'cross_w_k', 'w_k_cross', 'w_k2', 'w_k_2'])
    cross_w_v = get_param(['cross_attn_w_v', 'cross_w_v', 'w_v_cross', 'w_v2', 'w_v_2'])
    cross_w_o = get_param(['cross_attn_w_o', 'cross_w_o', 'w_o_cross', 'w_o2', 'w_o_2'])
    cross_gamma = get_param(['cross_attn_gamma', 'cross_gamma', 'attn_gamma_2', 'gamma2', 'cross_gamma'])
    cross_beta = get_param(['cross_attn_beta', 'cross_beta', 'attn_beta_2', 'beta2', 'cross_beta'])

    cross_attn_output = decoder_layer_cross_attention_sublayer(
        attn_output,
        encoder_output,
        cross_w_q,
        cross_w_k,
        cross_w_v,
        cross_w_o,
        cross_gamma,
        cross_beta,
        num_heads,
        src_mask
    )

    # --- 3. Feed-Forward Sublayer ---
    w1 = get_param(['w1', 'ffn_w1'])
    b1 = get_param(['b1', 'ffn_b1'])
    w2 = get_param(['w2', 'ffn_w2'])
    b2 = get_param(['b2', 'ffn_b2'])
    ffn_gamma = get_param(['ffn_gamma', 'gamma3', 'ffn_norm_gamma'])
    ffn_beta = get_param(['ffn_beta', 'beta3', 'ffn_norm_beta'])

    return decoder_layer_feed_forward_sublayer(
        cross_attn_output,
        w1,
        b1,
        w2,
        b2,
        ffn_gamma,
        ffn_beta
    )

# Step 47 - stack_decoder_layers
def stack_decoder_layers(y, encoder_output, decoder_layer_params_list, num_heads, src_mask, tgt_mask):
    # TODO: sequentially apply each decoder layer to the running target hidden state.
    
    for layer_params in decoder_layer_params_list:
       y = assemble_decoder_layer(y, encoder_output, layer_params, num_heads, src_mask, tgt_mask)

    return y

# Step 48 - apply_final_output_projection
def apply_final_output_projection(decoder_output, output_projection_weight, output_projection_bias=None):
    # TODO: project decoder hidden states (B, T, D) to vocabulary logits (B, T, V).
    return apply_linear_projection(decoder_output, output_projection_weight, output_projection_bias)

# Step 49 - tie_output_projection_to_token_embeddings
import torch

def tie_output_projection_to_token_embeddings(token_embedding_weight):
    """Return an output projection weight that shares storage with token_embedding_weight.

    Input shape: (vocab_size, d_model). Output shape: (d_model, vocab_size).
    """
    # TODO: return an output projection weight tied to the token embedding matrix
    
    return token_embedding_weight.T

# Step 50 - apply_log_softmax_over_vocab
import torch.nn as nn
def apply_log_softmax_over_vocab(logits):
    # TODO: Convert decoder logits (B, T, V) into log probabilities over the vocabulary axis.
    s =  nn.LogSoftmax(dim= -1)

    return s(logits)

# Step 51 - run_transformer_forward
def run_transformer_forward(src_ids, tgt_ids, model_params, num_heads, pad_id):
    # TODO: embed src+tgt, add PE, build masks, run encoder/decoder, project to log probs.
    
    # 1. Source and Target Embeddings
    token_emb_weights = model_params['token_embedding']
    d_model = token_emb_weights.shape[1]

    src_emb = F.embedding(src_ids,token_emb_weights)
    tgt_emb = F.embedding(tgt_ids , token_emb_weights)

    src_emb = scale_embeddings_by_sqrt_d_model(src_emb, d_model)
    tgt_emb = scale_embeddings_by_sqrt_d_model(tgt_emb, d_model)


    # 2. Positional Encoding
    max_len = max(src_ids.shape[1] ,tgt_ids.shape[1])
    pe = build_sinusoidal_positional_encoding(max_len, d_model)

    src_emb = add_positional_encoding_to_embeddings(src_emb, pe)
    tgt_emb = add_positional_encoding_to_embeddings(tgt_emb, pe)

    # 3. Build Masks

    src_mask = build_padding_mask(src_ids, pad_id)

    tgt_pad_mask = build_padding_mask(tgt_ids, pad_id)
    tgt_cas_mask = build_causal_mask(tgt_ids.shape[1])

    tgt_mask = combine_padding_and_causal_masks(tgt_pad_mask, tgt_cas_mask)

    # 4. Encoder

    encoder_out = stack_encoder_layers(
    src_emb, 
    model_params['encoder_layers'], 
    num_heads, 
    src_mask
    )

    # 5. Decoder

    decoder_out = stack_decoder_layers(
    tgt_emb, 
    encoder_out, 
    model_params['decoder_layers'], 
    num_heads, 
    src_mask, 
    tgt_mask
    )

    # 6. Output Projection and Log softmax

    output_projection_weights = model_params['output_projection']
    output_projection_bias = model_params.get('output_projection_bias' , None)

    logits = apply_final_output_projection(
    decoder_out, 
    output_projection_weights, 
    output_projection_bias)

    logits =  apply_log_softmax_over_vocab(logits)

    return logits

# Step 52 - init_encoder_layer_parameters
import torch
import math

def init_encoder_layer_parameters(d_model, num_heads, d_ff):
    """Return a dict of leaf tensors with requires_grad=True for one encoder layer."""
    # TODO: allocate w_q, w_k, w_v, w_o, w1, b1, w2, b2, attn_gamma, attn_beta, ffn_gamma, ffn_beta.
    params = {}

    def xavier_uniform(shape):
        fan_in , fan_out = shape[0] , shape[1]
        bound = math.sqrt(6 / (fan_in + fan_out))
        t = torch.empty(shape).uniform_(-bound , bound)
        t.requires_grad_(True)
        return t

    params['w_q'] = xavier_uniform((d_model , d_model))
    params['w_k'] = xavier_uniform((d_model , d_model))
    params['w_v'] = xavier_uniform((d_model , d_model))
    params['w_o'] = xavier_uniform((d_model , d_model))

    params['attn_gamma'] = torch.ones(d_model , requires_grad=True)
    params['attn_beta'] = torch.zeros(d_model , requires_grad=True)

    params['w1'] = xavier_uniform((d_model , d_ff))
    params['b1'] = torch.zeros(d_ff , requires_grad=True)
    
    params['w2'] = xavier_uniform((d_ff , d_model))
    params['b2'] = torch.zeros(d_model , requires_grad=True)

    params['ffn_gamma'] = torch.ones(d_model , requires_grad=True)
    params['ffn_beta'] = torch.zeros(d_model , requires_grad=True)

    return params

# Step 53 - init_decoder_layer_parameters
import torch
import math

def init_decoder_layer_parameters(d_model, num_heads, d_ff):
    # TODO: return a dict of requires_grad tensors for one decoder layer
    
    params = {}

    def xavier_uniform(shape):
        fan_in , fan_out = shape[0] ,shape[1]
        bound = math.sqrt(6 / (fan_in + fan_out))
        t = torch.empty(shape).uniform_(-bound , bound)
        t.requires_grad_(True)
        return t
    
    params['w_q_self'] = xavier_uniform((d_model ,d_model))
    params['w_k_self'] = xavier_uniform((d_model ,d_model))
    params['w_v_self'] = xavier_uniform((d_model ,d_model))
    params['w_o_self'] = xavier_uniform((d_model ,d_model))
    params['self_gamma'] = torch.ones(d_model , requires_grad = True)
    params['self_beta'] = torch.zeros(d_model , requires_grad = True)

    params['w_q_cross'] = xavier_uniform((d_model ,d_model))
    params['w_k_cross'] = xavier_uniform((d_model ,d_model))
    params['w_v_cross'] = xavier_uniform((d_model ,d_model))
    params['w_o_cross'] = xavier_uniform((d_model ,d_model))
    params['cross_gamma'] = torch.ones(d_model , requires_grad = True)
    params['cross_beta'] = torch.zeros(d_model , requires_grad = True)

    params['w1'] = xavier_uniform((d_model ,d_ff))
    params['b1'] = torch.zeros(d_ff , requires_grad = True)
    params['w2'] = xavier_uniform((d_ff ,d_model))
    params['b2'] = torch.zeros(d_model , requires_grad = True)
    params['ffn_gamma'] = torch.ones(d_model , requires_grad = True)
    params['ffn_beta'] = torch.zeros(d_model , requires_grad = True)

    return params

# Step 54 - init_embedding_and_projection_parameters
import torch
import math

def init_embedding_and_projection_parameters(vocab_size, d_model, tie_weights=True):
    """Allocate src/tgt embeddings and output projection (optionally tied)."""
    # TODO: allocate three (vocab_size, d_model) tensors with requires_grad=True
    
    def xavier_uniform(shape):
        fan_in , fan_out = shape[0] , shape[1]
        bound = math.sqrt(6 / (fan_in + fan_out))
        t = torch.empty(shape).uniform_(-bound , bound)
        t.requires_grad_(True)
        return t

    src_embedding = xavier_uniform((vocab_size ,d_model))
    tgt_embedding = xavier_uniform((vocab_size ,d_model))
    
    if tie_weights:
        output_projection = tgt_embedding
    else:
        output_projection = xavier_uniform((vocab_size , d_model))

    
    return {
        'src_embedding':src_embedding,
        'tgt_embedding':tgt_embedding,
        'output_projection':output_projection
    }

# Step 55 - collect_model_parameters_into_list
import torch

def collect_model_parameters_into_list(encoder_layer_params, decoder_layer_params, embedding_params):
    # TODO: walk the encoder, decoder, and embedding dicts and return a flat deduped list of tensors
    params_list = []
    seen = set()

    def add_params(tensor):
        if isinstance(tensor , torch.Tensor) and tensor.requires_grad:
            obj_id = id(tensor)
            if obj_id not in seen:
                seen.add(obj_id)
                params_list.append(tensor)
    
    for layer in encoder_layer_params:
        for val in layer.values():
            add_params(val)

    for layer in decoder_layer_params:
        for val in layer.values():
            add_params(val)

    for val in embedding_params.values():
            add_params(val)

    return params_list

# Step 56 - shift_targets_right_with_start_token
import torch
def shift_targets_right_with_start_token(target_ids, start_token_id):
    # TODO: prepend start_token_id and drop the last column so output shape matches target_ids
    B,L = target_ids.shape

    col = torch.full((B,1), start_token_id, dtype = target_ids.dtype , device = target_ids.device)

    return torch.cat([col ,target_ids[:,:-1]],dim = 1)

# Step 57 - compute_noam_learning_rate
def compute_noam_learning_rate(step, d_model, warmup_steps):
    # TODO: return the Noam warmup learning rate for the given step.
    return d_model ** (-0.5) * min(step**(-0.5) , step*(warmup_steps ** (-1.5)))

# Step 58 - build_uniform_smoothing_distribution
import torch

def build_uniform_smoothing_distribution(shape, vocab_size, epsilon):
    # TODO: return a float tensor of `shape` filled with epsilon / (vocab_size - 2).
    return torch.full((shape) , (epsilon/(vocab_size - 2)) , dtype = torch.float32)

# Step 59 - set_confidence_on_gold_tokens
import torch

def set_confidence_on_gold_tokens(smoothed_distribution, gold_token_ids, confidence):
    """Place confidence mass at gold-token positions of a smoothed target distribution."""
    # TODO: write the confidence value at each gold token id along the vocab axis
    index = gold_token_ids.unsqueeze(-1)

    return smoothed_distribution.scatter(-1,index,confidence)

# Step 60 - zero_pad_column_and_pad_token_rows
import torch

def zero_pad_column_and_pad_token_rows(smoothed_distribution, gold_token_ids, pad_id):
    # TODO: zero the pad column and the rows where the gold token equals pad_id
    smoothed_distribution[...,pad_id] = 0.0

    is_pad_row = (gold_token_ids == pad_id).unsqueeze(-1)

    return torch.where(is_pad_row , torch.zeros_like(smoothed_distribution) , smoothed_distribution)

# Step 61 - compute_label_smoothed_kl_loss
import torch

def compute_label_smoothed_kl_loss(log_probabilities, smoothed_distribution):
    """Return the summed KL loss over all (batch, time, vocab) entries."""
    # TODO: combine log_probabilities with the smoothed target distribution into a scalar loss
    return abs(torch.sum(log_probabilities * smoothed_distribution))

# Step 62 - average_loss_over_non_pad_tokens
import torch

def average_loss_over_non_pad_tokens(total_loss, gold_token_ids, pad_id):
    # TODO: divide total_loss by the count of non-pad tokens in gold_token_ids
    non_pad_mask = (gold_token_ids!=pad_id)

    num_non_pad = non_pad_mask.sum().to(dtype = total_loss.dtype)

    denominator = torch.clamp(num_non_pad , min = 1.0)

    return total_loss / denominator

# Step 63 - compute_token_accuracy_ignoring_pad
import torch

def compute_token_accuracy_ignoring_pad(log_probabilities, gold_token_ids, pad_id):
    # TODO: argmax over vocab, compare to gold, average over non-pad positions only
    predictions = torch.argmax(log_probabilities , dim = -1)

    non_pad_mask = (gold_token_ids != pad_id)

    num_non_pad = non_pad_mask.sum().to(dtype = log_probabilities.dtype , device = log_probabilities.device)

    if num_non_pad == 0:
        return torch.tensor(0.0 , dtype = log_probabilities.dtype ,device = log_probabilities.device)
    
    correct_mask = (predictions == gold_token_ids) & non_pad_mask
    num_correct = correct_mask.sum().to(dtype = torch.float32)

    return num_correct / num_non_pad

# Step 64 - initialize_adam_optimizer_state
import torch

def initialize_adam_optimizer_state(parameter_list):
    """Allocate Adam m, v zero buffers and a step counter t=0."""
    # TODO: allocate zero buffers for first and second moments, plus step counter
    return {
        'm':[torch.zeros_like(p) for p in parameter_list] ,
        'v':[torch.zeros_like(p) for p in parameter_list] ,
        't':0
    }

# Step 65 - update_adam_first_moment
import torch

def update_adam_first_moment(m_prev, grad, beta1):
    """Return m_t = beta1 * m_prev + (1 - beta1) * grad."""
    # TODO: apply the Adam first-moment EMA update and return the new tensor
    m_t = beta1 * m_prev + (1-beta1) * grad
    return m_t

# Step 66 - update_adam_second_moment
import torch

def update_adam_second_moment(v_prev, grad, beta2):
    """Return v_t = beta2 * v_prev + (1 - beta2) * grad ** 2."""
    # TODO: apply Adam's EMA update for the second moment of the gradient
    v_t = beta2 * v_prev + (1-beta2) * grad ** 2
    return v_t

# Step 67 - apply_adam_bias_correction
import torch

def apply_adam_bias_correction(m_t, v_t, beta1, beta2, step):
    """Return bias-corrected (m_hat, v_hat) for Adam at the given step."""
    # TODO: divide each moment by (1 - beta**step) using its respective beta
    m_hat = (m_t / (1 - (beta1 ** step)))

    v_hat = (v_t / (1 - (beta2 ** step)))

    return (m_hat , v_hat)

# Step 69 - apply_adam_step_to_all_parameters
import torch

def apply_adam_step_to_all_parameters(parameter_list, optimizer_state, learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9):
    
    optimizer_state['t'] += 1
    t = optimizer_state['t']

    with torch.no_grad():
        for i, p in enumerate(parameter_list):
            if p.grad is None:
                continue

            grad = p.grad

            
            optimizer_state['m'][i] = update_adam_first_moment(optimizer_state['m'][i], grad, beta1)
            optimizer_state['v'][i] = update_adam_second_moment(optimizer_state['v'][i], grad, beta2)

            
            m_hat, v_hat = apply_adam_bias_correction(
                optimizer_state['m'][i], 
                optimizer_state['v'][i], 
                beta1, 
                beta2, 
                t
            )

            
            p.addcdiv_(m_hat, torch.sqrt(v_hat) + epsilon, value=-learning_rate)

    return optimizer_state

# Step 70 - zero_all_parameter_gradients
import torch

def zero_all_parameter_gradients(parameter_list):
    """Clear the .grad of every parameter tensor before the next backward pass."""
    # TODO: clear the accumulated gradient on every parameter tensor in the list
    for p in parameter_list:
        p.grad = None

# Step 71 - compute_batch_training_loss
def compute_batch_training_loss(src_batch, tgt_batch, model_params, config):
    # TODO: shift targets right, run the forward pass, build smoothed targets, and average the KL loss over non-pad tokens.
    pad_id = config['pad_id']
    start_id = config['start_id']
    vocab_size = config['vocab_size']
    num_heads = config['num_heads']
    smoothing = config['smoothing']


    decoder_input = shift_targets_right_with_start_token(tgt_batch, start_id)

    log_probas = run_transformer_forward(src_batch, decoder_input, model_params, num_heads, pad_id)

    confidence = 1 - smoothing

    smoothned_dist = build_uniform_smoothing_distribution(log_probas.shape, vocab_size, confidence)
    smoothned_dist = set_confidence_on_gold_tokens(smoothned_dist, tgt_batch, confidence)
    smoothned_dist = zero_pad_column_and_pad_token_rows(smoothned_dist, tgt_batch, pad_id)
    
    
    total_loss = compute_label_smoothed_kl_loss(log_probas, smoothned_dist)
    average_loss = average_loss_over_non_pad_tokens(total_loss, tgt_batch, pad_id)


    return average_loss

# Step 72 - run_training_step_with_backprop (not yet solved)
# TODO: implement

# Step 73 - run_training_loop_for_steps (not yet solved)
# TODO: implement

# Step 74 - pick_next_token_by_argmax (not yet solved)
# TODO: implement

# Step 75 - compute_length_penalty (not yet solved)
# TODO: implement

# Step 76 - compute_candidate_scores (not yet solved)
# TODO: implement

# Step 77 - select_top_k_candidates (not yet solved)
# TODO: implement

# Step 78 - append_tokens_to_beam_sequences (not yet solved)
# TODO: implement

# Step 79 - mark_finished_beams (not yet solved)
# TODO: implement

# Step 80 - select_best_finished_beam (not yet solved)
# TODO: implement

