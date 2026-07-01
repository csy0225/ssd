"""Iluvatar BI-V150 (天数) attention backend for SSD.

Neither `sgl_kernel` nor `flashinfer` builds exist on the Iluvatar CoreX
stack. Prefill uses the shipped FlashAttention-2 (`flash_attn` 2.6.3); the
paged decode / verify / async-tree paths gather KV densely and run Iluvatar's
fused masked attention `ixformer.ixinfer_flash_attn_pad` (arbitrary float
additive mask, GQA-native), falling back to torch SDPA if ixformer is absent.

KV cache layout is flashinfer's "NHD": [num_blocks, block_size, num_kv_heads,
head_dim]. The decode/verify paths are static-shape (no `.item()`/loops) so
they can be captured by torch.cuda.graph.
"""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func

try:
    from ixformer import ixinfer_flash_attn_pad as _ix_attn
    _HAS_IX = True
except Exception:
    _ix_attn = None
    _HAS_IX = False


def _masked_attn(q, k, v, add_mask, scale):
    """Attention with an additive mask. q:[B,Hq,S,D] k/v:[B,Hkv,T,D]
    add_mask:[B,1,S,T] float32 (0 keep / -inf block) or None. GQA-native."""
    if _HAS_IX:
        if add_mask is None:
            add_mask = torch.zeros(q.shape[0], 1, q.shape[2], k.shape[2],
                                   dtype=torch.float32, device=q.device)
        return _ix_attn(q.contiguous(), k.contiguous(), v.contiguous(),
                        add_mask.to(torch.float32), atten_scale=scale)
    # torch SDPA fallback: expand GQA + cast mask to q dtype
    rep = q.shape[1] // k.shape[1]
    if rep > 1:
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    m = add_mask.to(q.dtype) if add_mask is not None else None
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m, scale=scale)


def _seqlens_from_indptr(indptr: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Per-segment lengths and max length from a CSR-style indptr tensor."""
    lens = (indptr[1:] - indptr[:-1]).to(torch.int32)
    return lens, int(lens.max().item()) if lens.numel() else 0


def prefill_ragged(q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
    """Varlen non-paged causal attention (prefill)."""
    cu_q = cu_seqlens_q.to(torch.int32)
    cu_k = cu_seqlens_k.to(torch.int32)
    _, max_q = _seqlens_from_indptr(cu_q)
    _, max_k = _seqlens_from_indptr(cu_k)
    return flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_q, cu_seqlens_k=cu_k,
        max_seqlen_q=max_q, max_seqlen_k=max_k,
        softmax_scale=scale, causal=True,
    )


def _gather_paged_kv(cache, block_tables_row, ctx_len, block_size, num_kv_heads, head_dim):
    """Gather a single sequence's contiguous KV [ctx_len, n_kv, hd]."""
    num_pages = (ctx_len + block_size - 1) // block_size
    pages = block_tables_row[:num_pages]
    gathered = cache[pages].reshape(num_pages * block_size, num_kv_heads, head_dim)
    return gathered[:ctx_len]


def paged_attn(q, k_cache, v_cache, block_tables, context_lens, qo_indptr, scale):
    """Paged causal attention (variable per-seq q-len fallback; eager).

    q: [total_q_tokens, num_qo_heads, head_dim]. qo_indptr marks per-request
    query boundaries. Tail-aligned causal mask (query i at abs pos ctx-q_len+i).
    """
    qo_indptr = qo_indptr.to(torch.int32)
    block_tables = block_tables.to(torch.int64)
    context_lens = context_lens.to(torch.int64)
    block_size = k_cache.shape[1]
    n_kv, hd = k_cache.shape[2], k_cache.shape[3]
    nq = q.shape[1]
    bs = qo_indptr.numel() - 1

    outs = []
    for b in range(bs):
        qs, qe = int(qo_indptr[b]), int(qo_indptr[b + 1])
        q_len = qe - qs
        ctx_len = int(context_lens[b])
        k_seq = _gather_paged_kv(k_cache, block_tables[b], ctx_len, block_size, n_kv, hd)
        v_seq = _gather_paged_kv(v_cache, block_tables[b], ctx_len, block_size, n_kv, hd)

        q_h = q[qs:qe].transpose(0, 1).unsqueeze(0)            # [1, nq, q_len, hd]
        k_h = k_seq.transpose(0, 1).unsqueeze(0)               # [1, n_kv, ctx, hd]
        v_h = v_seq.transpose(0, 1).unsqueeze(0)
        rows = torch.arange(q_len, device=q.device).unsqueeze(1)
        cols = torch.arange(ctx_len, device=q.device).unsqueeze(0)
        allowed = cols <= (ctx_len - q_len + rows)             # [q_len, ctx]
        mask = torch.zeros(1, 1, q_len, ctx_len, dtype=torch.float32, device=q.device)
        mask.masked_fill_(~allowed[None, None], float("-inf"))
        o_h = _masked_attn(q_h, k_h, v_h, mask, scale).squeeze(0)   # [nq, q_len, hd]
        outs.append(o_h.transpose(0, 1).reshape(q_len, nq, hd))
    return torch.cat(outs, dim=0)


def paged_decode(q, k_cache, v_cache, block_tables, context_lens, scale):
    """CUDA-graph-safe single-query decode (one query token per seq).

    Static-shape: gather ALL max_blocks per seq, mask positions >= context_lens.
    q: [bs, nq, hd]; block_tables: [bs, max_blocks]; context_lens: [bs].
    """
    bs, nq, hd = q.shape
    max_blocks = block_tables.shape[1]
    block_size = k_cache.shape[1]
    n_kv = k_cache.shape[2]
    max_ctx = max_blocks * block_size

    bt = block_tables.clamp_min(0).to(torch.long)              # [bs, max_blocks]
    kg = k_cache[bt].reshape(bs, max_ctx, n_kv, hd).permute(0, 2, 1, 3)   # [bs, n_kv, ctx, hd]
    vg = v_cache[bt].reshape(bs, max_ctx, n_kv, hd).permute(0, 2, 1, 3)
    qh = q.unsqueeze(2)                                        # [bs, nq, 1, hd]

    pos = torch.arange(max_ctx, device=q.device)
    allowed = pos[None, :] < context_lens.to(torch.long)[:, None]   # [bs, ctx]
    mask = torch.zeros(bs, 1, 1, max_ctx, dtype=torch.float32, device=q.device)
    mask.masked_fill_(~allowed[:, None, None, :], float("-inf"))

    o = _masked_attn(qh, kg, vg, mask, scale)                 # [bs, nq, 1, hd]
    return o.squeeze(2).reshape(bs, nq, hd)


def paged_verify(q, k_cache, v_cache, block_tables, context_lens, q_len, scale):
    """CUDA-graph-safe multi-query verify (uniform q_len=K+1 per seq).

    Static-shape: gather all max_blocks, tail-aligned causal mask.
    q: [bs*q_len, nq, hd]; block_tables [bs, max_blocks]; context_lens [bs].
    """
    total, nq, hd = q.shape
    bs = total // q_len
    max_blocks = block_tables.shape[1]
    block_size = k_cache.shape[1]
    n_kv = k_cache.shape[2]
    max_ctx = max_blocks * block_size

    bt = block_tables.clamp_min(0).to(torch.long)
    kg = k_cache[bt].reshape(bs, max_ctx, n_kv, hd).permute(0, 2, 1, 3)
    vg = v_cache[bt].reshape(bs, max_ctx, n_kv, hd).permute(0, 2, 1, 3)
    qh = q.reshape(bs, q_len, nq, hd).permute(0, 2, 1, 3)      # [bs, nq, q_len, hd]

    pos = torch.arange(max_ctx, device=q.device)
    qi = torch.arange(q_len, device=q.device)
    cl = context_lens.to(torch.long)
    absq = cl[:, None] - q_len + qi[None, :]                   # [bs, q_len]
    allowed = pos[None, None, :] <= absq[:, :, None]           # [bs, q_len, ctx]
    mask = torch.zeros(bs, 1, q_len, max_ctx, dtype=torch.float32, device=q.device)
    mask.masked_fill_(~allowed[:, None, :, :], float("-inf"))

    o = _masked_attn(qh, kg, vg, mask, scale)                 # [bs, nq, q_len, hd]
    return o.permute(0, 2, 1, 3).reshape(total, nq, hd)


class EagerTreeAttn:
    """Async tree-decode attention (replaces flashinfer paged wrapper). Runs
    the packed tree mask through the fused ixinfer kernel (SDPA fallback)."""

    def __init__(self, *_, **__):
        self._planned = False

    def plan(self, qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
             num_qo_heads, num_kv_heads, head_dim, block_size,
             custom_mask=None, q_data_type=None, kv_data_type=None, **_):
        self.qo_indptr = qo_indptr.to(torch.int32)
        self.kv_indptr = kv_indptr.to(torch.int32)
        self.kv_indices = kv_indices.to(torch.int32)
        self.kv_last_page_len = kv_last_page_len.to(torch.int32)
        self.num_qo_heads = int(num_qo_heads)
        self.num_kv_heads = int(num_kv_heads)
        self.head_dim = int(head_dim)
        self.block_size = int(block_size)
        self.custom_mask = custom_mask  # flat bool, row-major [MQ_b, ctx_b] per seq
        self._planned = True

    def _gather_seq_kv(self, cache, seq_pages, ctx_len):
        gathered = cache[seq_pages].reshape(-1, self.num_kv_heads, self.head_dim)
        return gathered[:ctx_len]

    def run(self, q, kv):
        assert self._planned, "EagerTreeAttn.run called before plan"
        k_cache, v_cache = kv
        qo_indptr = self.qo_indptr
        kv_indptr = self.kv_indptr
        bs = qo_indptr.numel() - 1
        scale = self.head_dim ** -0.5

        mask_off = 0
        outs = []
        for b in range(bs):
            qs, qe = int(qo_indptr[b]), int(qo_indptr[b + 1])
            mq = qe - qs
            ps, pe = int(kv_indptr[b]), int(kv_indptr[b + 1])
            seq_pages = self.kv_indices[ps:pe]
            n_pages = pe - ps
            ctx_len = (n_pages - 1) * self.block_size + int(self.kv_last_page_len[b])

            k_seq = self._gather_seq_kv(k_cache, seq_pages, ctx_len)   # [ctx, n_kv, hd]
            v_seq = self._gather_seq_kv(v_cache, seq_pages, ctx_len)

            q_h = q[qs:qe].transpose(0, 1).unsqueeze(0)                # [1, n_qo, mq, hd]
            k_h = k_seq.transpose(0, 1).unsqueeze(0)                   # [1, n_kv, ctx, hd]
            v_h = v_seq.transpose(0, 1).unsqueeze(0)

            n = mq * ctx_len
            keep = self.custom_mask[mask_off:mask_off + n].view(mq, ctx_len).to(torch.bool)
            mask_off += n
            mask = torch.zeros(1, 1, mq, ctx_len, dtype=torch.float32, device=q.device)
            mask.masked_fill_(~keep[None, None], float("-inf"))

            o_h = _masked_attn(q_h, k_h, v_h, mask, scale).squeeze(0)  # [n_qo, mq, hd]
            outs.append(o_h.transpose(0, 1).reshape(mq, self.num_qo_heads, self.head_dim))
        return torch.cat(outs, dim=0)
