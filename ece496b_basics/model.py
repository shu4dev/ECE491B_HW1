import os
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
from typing import IO, BinaryIO, Optional
import numpy.typing as npt

class RMSnorm(nn.Module):
    def __init__(self, d_model: int, eps: float=1e-5):
        super(RMSnorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x / torch.sqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps) * self.weight

class GELU(nn.Module):
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return x * 0.5 * (1.0 + torch.erf(x / np.sqrt(2.0)))

class positionwise_feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super(positionwise_feedforward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.gelu = GELU()
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        l0 = self.gelu(self.w1(x))
        l1 = self.w2(l0)
        return l1

class softmax(nn.Module):
    def __init__(self, dim: int):
        super(softmax, self).__init__()
        self.dim = dim
    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        shift_x = x - torch.max(x, dim=self.dim, keepdim=True)[0]
        exp_x = torch.exp(shift_x)
        sum_exp = exp_x.sum(dim=self.dim, keepdim=True)
        return exp_x / sum_exp

def run_scaled_dot_product_attention(
    K: torch.FloatTensor,
    Q: torch.FloatTensor,
    V: torch.FloatTensor,
    mask: Optional[torch.BoolTensor] = None,
    pdrop: Optional[float] = None,
) -> torch.FloatTensor:

    d_k = K.shape[-1]
    attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
    my_softmax = softmax(dim=-1)
    attn_weights = my_softmax(attn_scores)
    attn_weights = F.dropout(attn_weights, p=pdrop)
    return torch.matmul(attn_weights, V)

class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        super(MultiheadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.attn_pdrop = attn_pdrop
        self.d_k = d_model // num_heads
        self.d_v = self.d_k
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
    
    def load_state_dict(self, weights):
        self.q_proj.weight.data = torch.cat([weights[f"q_heads.{i}.weight"] for i in range(self.num_heads)])
        self.k_proj.weight.data = torch.cat([weights[f"k_heads.{i}.weight"] for i in range(self.num_heads)])
        self.v_proj.weight.data = torch.cat([weights[f"v_heads.{i}.weight"] for i in range(self.num_heads)])
        self.output_proj.weight.data = weights['output_proj.weight']

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        batch_size, seq_len, _ = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_v).transpose(1, 2)

        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(x.device)
        context = run_scaled_dot_product_attention(K, Q, V, mask=mask, pdrop=self.attn_pdrop)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.output_proj(context)

class transformer_block(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, attn_pdrop: float, residual_pdrop: float):
        super(transformer_block, self).__init__()
        self.ln1 = RMSnorm(d_model)
        self.ln2 = RMSnorm(d_model)
        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_pdrop)
        self.ffn = positionwise_feedforward(d_model, d_ff)
        self.drop1 = nn.Dropout(residual_pdrop)
        self.drop2 = nn.Dropout(residual_pdrop)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        x = x + self.drop1(self.attn(self.ln1(x)))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x

class transformer_lm(nn.Module):
    def __init__(self, 
    vocab_size: int, 
    context_length: int, 
    d_model: int, 
    num_layers: int,
    num_heads: int,
    d_ff: int,
    attn_pdrop: float,
    residual_pdrop: float):
        
        super(transformer_lm, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(context_length, d_model)
        self.layers = nn.ModuleList([transformer_block(d_model, num_heads, d_ff, attn_pdrop, residual_pdrop) for _ in range(num_layers)])
        self.ln_final = RMSnorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.residual_pdrop = residual_pdrop
    def forward(self, x: torch.LongTensor) -> torch.FloatTensor:
        batch_size, seq_len = x.shape
        positions = torch.arange(seq_len, device=x.device).expand(batch_size, seq_len)
        x = self.token_embeddings(x) + self.position_embeddings(positions)
        x = F.dropout(x, p = self.residual_pdrop)
        for block in self.layers:
            x = block(x)
        x = self.ln_final(x)
        return self.lm_head(x)

class cross_entropy_loss(nn.Module):
    def __init__(self):
        super(cross_entropy_loss, self).__init__()

    def forward(self, inputs: torch.FloatTensor, targets: torch.LongTensor) -> torch.FloatTensor:
        log_softmax = inputs - torch.logsumexp(inputs, dim=-1, keepdim=True)
        log_probs = torch.gather(log_softmax, dim=-1, index=targets.unsqueeze(-1))
        loss = -log_probs.mean()
        return loss
 
def get_lr_cosine_schedule(t, lr_max, lr_min, warmup_iters, total_iters, **kwargs):
    if t < warmup_iters:
        return lr_max * t / warmup_iters
    elif t < total_iters:
        return lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos((t - warmup_iters) / (total_iters - warmup_iters) * 3.141592653589793))
    else:
        return lr_min

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=None, weight_decay=0.001, betas=(0.9, 0.999), eps=1e-8, **kwargs):
        defaults = dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        super(AdamW, self).__init__(params, defaults)
    
    def set_lr(self, lr):
        for group in self.param_groups:
            group['lr'] = lr
    
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            weight_decay = group['weight_decay']
            beta1, beta2 = group['betas']
            eps = group['eps']
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                t = state['t'] + 1
                m, v = state['m'], state['v']
                m = beta1 * m + (1 - beta1) * grad
                v = beta2 * v + (1 - beta2) * grad ** 2
                lr_t = lr * (1 - beta2 ** t) ** 0.5 / (1 - beta1 ** t) 
                p.data -= lr_t * m / (v ** 0.5 + eps)
                p.data -= lr * weight_decay * p.data
                state['t'] = t
                state['m'] = m
                state['v'] = v

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    starting_indices = torch.randint(0, len(dataset) - context_length, (batch_size,))
    x = torch.stack([torch.from_numpy(dataset[start_idx:start_idx + context_length]) for start_idx in starting_indices])
    y = torch.stack([torch.from_numpy(dataset[start_idx + 1:start_idx + context_length + 1]) for start_idx in starting_indices])
    return x.to(device).long(), y.to(device).long()

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    iteration = checkpoint['iteration']
    return iteration

def gradient_clipping(parameters, max_norm):
    # Only consider parameters with gradients
    total_norm_2 = sum([torch.sum(p.grad ** 2) for p in parameters if p.grad is not None])
    total_norm = total_norm_2 ** 0.5
    if total_norm > max_norm:
        for p in parameters:
            if p.grad is not None:
                p.grad.detach().mul_(max_norm / total_norm)