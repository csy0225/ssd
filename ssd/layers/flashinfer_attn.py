"""FlashInfer-based implementations of the three attention ops that SSD's
`attention.py` normally takes from `sgl_kernel.flash_attn`.

Used as a compile-free fallback when `sgl_kernel` is not installed. The KV
cache layout (`[num_blocks, block_size, num_kv_heads, head_dim]`, "NHD") matches
what SSD already feeds to its flashinfer tree-attention wrappers.
"""
import torch
import flashinfer

_WS_BYTES = 256 * 1024 * 1024
_ragged_wrappers = {}
_paged_wrappers = {}


def _ragged(device):
    if device not in _ragged_wrappers:
        ws = torch.empty(_WS_BYTES, dtype=torch.uint8, device=device)
        _ragged_wrappers[device] = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(ws, "NHD")
    return _ragged_wrappers[device]


def _paged(device):
    if device not in _paged_wrappers:
        ws = torch.empty(_WS_BYTES, dtype=torch.uint8, device=device)
        _paged_wrappers[device] = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws, "NHD")
    return _paged_wrappers[device]


def prefill_ragged(q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
    """Varlen non-paged causal attention (replaces flash_attn_varlen_func)."""
    w = _ragged(q.device)
    w.plan(
        cu_seqlens_q.to(torch.int32),
        cu_seqlens_k.to(torch.int32),
        q.shape[1],          # num_qo_heads
        k.shape[1],          # num_kv_heads
        q.shape[2],          # head_dim_qk
        causal=True,
        sm_scale=scale,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    return w.run(q, k, v)


def _build_paged_indices(block_tables, context_lens, block_size):
    dev = block_tables.device
    ctx = context_lens.to(torch.int32)
    bs = ctx.shape[0]
    num_pages = (ctx + block_size - 1) // block_size          # [bs]
    kv_indptr = torch.zeros(bs + 1, dtype=torch.int32, device=dev)
    kv_indptr[1:] = torch.cumsum(num_pages, 0).to(torch.int32)
    max_pages = block_tables.shape[1]
    valid = torch.arange(max_pages, device=dev)[None, :] < num_pages[:, None]
    kv_indices = block_tables[valid].to(torch.int32)
    last_page_len = ctx - (num_pages - 1) * block_size         # [bs], in [1, block_size]
    return kv_indptr, kv_indices, last_page_len.to(torch.int32)


def paged_attn(q, k_cache, v_cache, block_tables, context_lens, qo_indptr, scale):
    """Paged causal attention (replaces flash_attn_with_kvcache).

    q: [total_q_tokens, num_qo_heads, head_dim]. qo_indptr marks per-request
    query boundaries (single-query decode -> arange(bs+1); verify -> cu_seqlens_q).
    """
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    head_dim = k_cache.shape[3]
    kv_indptr, kv_indices, last_page_len = _build_paged_indices(
        block_tables, context_lens, block_size)
    w = _paged(q.device)
    w.plan(
        qo_indptr.to(torch.int32),
        kv_indptr,
        kv_indices,
        last_page_len,
        q.shape[1],          # num_qo_heads
        num_kv_heads,
        head_dim,
        block_size,
        causal=True,
        sm_scale=scale,
        q_data_type=q.dtype,
        kv_data_type=k_cache.dtype,
    )
    return w.run(q, (k_cache, v_cache))
