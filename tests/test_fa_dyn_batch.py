from dataclasses import dataclass
from typing import Tuple
import torch

"""
Gold implementation is based on LLama3 (https://github.com/meta-llama/llama3/blob/main/llama/model.py)

"""


def rms_norm(x: torch.Tensor, eps=1e-05):
    # x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    norm = x.pow(2)  # map
    norm = norm.mean(-1, keepdim=True)  # accum & repeat & map
    norm = torch.rsqrt(norm + eps)  # map
    return x * norm  # map


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def gold_calc(
    x: torch.Tensor,
    model_config,
    wq: torch.nn.Module,
    wk: torch.nn.Module,
    wv: torch.nn.Module,
    wo: torch.nn.Module,
):
    # x: [batch, 1, hidden_dim]
    # wq: [hidden_dim, hidden_dim]
    # wk: [hidden_dim, hidden_dim]
    # wv: [hidden_dim, hidden_dim]
    # wo: [hidden_dim, hidden_dim]
    bsz, seqlen, _ = x.shape

    rms_norm_eps = 1e-05

    # ====== RMS Norm ======
    x = rms_norm(x, rms_norm_eps)

    # ====== Attention ======
    xq, xk, xv = wq(x), wk(x), wv(x)
    xq = xq.view(bsz, seqlen, model_config.n_heads, model_config.head_dim)
    xk = xk.view(bsz, seqlen, model_config.n_kv_heads, model_config.head_dim)
    xv = xv.view(bsz, seqlen, model_config.n_kv_heads, model_config.head_dim)

    # skip rotary emb for now
    # xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)
    cache_k = cache_k.to(xq)
    cache_v = cache_v.to(xq)

    cache_k[:bsz, start_pos : start_pos + seqlen] = xk
    cache_v[:bsz, start_pos : start_pos + seqlen] = xv

    keys = cache_k[:bsz, : start_pos + seqlen]
    values = cache_v[:bsz, : start_pos + seqlen]

    # repeat k/v heads if n_kv_heads < n_heads
    keys = repeat_kv(
        keys, model_config.n_rep
    )  # (bs, cache_len + seqlen, n_local_heads, head_dim)
    values = repeat_kv(
        values, model_config.n_rep
    )  # (bs, cache_len + seqlen, n_local_heads, head_dim)

    xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
    keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
    scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
    if mask is not None:
        scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
    scores = F.softmax(scores.float(), dim=-1).type_as(xq)
    output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
    output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
    return self.wo(output)


@dataclass
class ModelConfig:
    hidden_dim: 128  # 16 * 8
    n_heads: 8
    n_kv_heads: 2
    head_dim: 16


def fa_dyn_batch():

    # QKV generation

    #

    pass


def test_fa_dyn_batch():
    torch.manual_seed(0)

    model_config = ModelConfig(hidden_dim=2048, n_heads=16)

    batch = 16

    max_generation_len = 32
    seq_lens = [4, 5, 2, 32, 15, 13, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12]
    max_seq_len = max(seq_lens) + max_generation_len

    # weights
    wq = torch.nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
    wk = torch.nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
    wv = torch.nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)
    wo = torch.nn.Linear(model_config.hidden_dim, model_config.hidden_dim, bias=False)

    # input
    x = torch.randn(batch, 1, model_config.hidden_dim)

    # cache
    cache_k = torch.zeros(
        batch, max_seq_len, model_config.n_kv_heads, model_config.head_dim
    )
    cache_v = torch.zeros(
        batch, max_seq_len, model_config.n_kv_heads, model_config.head_dim
    )
    for i, seq_len in enumerate(seq_lens):
        cache_k[i, :seq_len, :, :] = torch.randn(
            seq_len, model_config.n_kv_heads, model_config.head_dim
        )
        cache_v[i, :seq_len, :, :] = torch.randn(
            seq_len, model_config.n_kv_heads, model_config.head_dim
        )

    # Process the KV cache for parallel processing
    cache_k_list: Tuple[torch.Tensor, ...] = torch.unbind(
        cache_k, dim=2
    )  # [batch, max_seq_len, head_dim] (len: n_kv_heads)
    cache_v_list: Tuple[torch.Tensor, ...] = torch.unbind(cache_v, dim=2)

    gold_calc(x, model_config, wq, wk, wv, wo)
