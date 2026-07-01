"""Iluvatar BI-V150 (天数) attention backend for SSD.

Neither `sgl_kernel` nor `flashinfer` builds exist on the Iluvatar CoreX
stack, but Iluvatar ships a standard FlashAttention-2 package (`flash_attn`
2.6.3) exposing `flash_attn_varlen_func` / `flash_attn_with_kvcache`. This
module maps SSD's three attention ops onto that API, plus a torch-SDPA
`EagerTreeAttn` wrapper that replaces the flashinfer paged wrapper used by the
async tree-decode path (FA2 has no arbitrary-mask kernel).

KV cache layout is flashinfer's "NHD": [num_blocks, block_size, num_kv_heads,
head_dim] — identical to what FA2 `flash_attn_with_kvcache(block_table=...)`
expects, so no relayout is needed.

Eager only: the SDPA tree gather is not cudagraph-safe (same limitation the
flashinfer fallback had).
"""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_varlen_func


def _seqlens_from_indptr(indptr: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Per-segment lengths and max length from a CSR-style indptr tensor."""
    lens = (indptr[1:] - indptr[:-1]).to(torch.int32)
    return lens, int(lens.max().item()) if lens.numel() else 0


def prefill_ragged(q, k, v, cu_seqlens_q, cu_seqlens_k, scale):
    """Varlen non-paged causal attention (replaces flash_attn_varlen_func)."""
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
    """Gather a single sequence's contiguous KV [ctx_len, n_kv, hd] from the
    paged cache [num_blocks, block_size, n_kv, hd]."""
    num_pages = (ctx_len + block_size - 1) // block_size
    pages = block_tables_row[:num_pages]                       # [num_pages]
    gathered = cache[pages]                                     # [num_pages, block_size, n_kv, hd]
    gathered = gathered.reshape(num_pages * block_size, num_kv_heads, head_dim)
    return gathered[:ctx_len]                                   # [ctx_len, n_kv, hd]


def paged_attn(q, k_cache, v_cache, block_tables, context_lens, qo_indptr, scale):
    """Paged causal attention (replaces flash_attn_with_kvcache).

    Iluvatar's flash_attn exposes flash_attn_with_kvcache but the underlying
    kvcache CUDA kernel raises "not supported", so we gather each sequence's KV
    from the paged cache and run torch SDPA with is_causal=True. The q tokens
    (1 for decode, K+1 for verify) align to the tail of the context, which is
    exactly the append-then-attend causal semantics.

    q: [total_q_tokens, num_qo_heads, head_dim]. qo_indptr marks per-request
    query boundaries (single-query decode -> arange(bs+1); verify -> cu_seqlens_q).
    """
    qo_indptr = qo_indptr.to(torch.int32)
    block_tables = block_tables.to(torch.int64)
    context_lens = context_lens.to(torch.int64)
    block_size = k_cache.shape[1]
    n_kv, hd = k_cache.shape[2], k_cache.shape[3]
    nq = q.shape[1]
    rep = nq // n_kv
    bs = qo_indptr.numel() - 1

    outs = []
    for b in range(bs):
        qs, qe = int(qo_indptr[b]), int(qo_indptr[b + 1])
        ctx_len = int(context_lens[b])
        k_seq = _gather_paged_kv(k_cache, block_tables[b], ctx_len, block_size, n_kv, hd)
        v_seq = _gather_paged_kv(v_cache, block_tables[b], ctx_len, block_size, n_kv, hd)

        q_h = q[qs:qe].transpose(0, 1)                          # [nq, q_len, hd]
        k_h = k_seq.transpose(0, 1).repeat_interleave(rep, dim=0)  # [nq, ctx, hd]
        v_h = v_seq.transpose(0, 1).repeat_interleave(rep, dim=0)
        q_len = qe - qs
        # Tail-aligned causal mask: query i sits at absolute position
        # (ctx_len - q_len + i) and attends to kv[0 .. that position]. torch's
        # is_causal uses a TOP-LEFT triangle (wrong when q_len < ctx_len), so we
        # build the offset mask explicitly.
        rows = torch.arange(q_len, device=q.device).unsqueeze(1)
        cols = torch.arange(ctx_len, device=q.device).unsqueeze(0)
        allowed = cols <= (ctx_len - q_len + rows)              # [q_len, ctx]
        attn_mask = torch.zeros(q_len, ctx_len, dtype=q_h.dtype, device=q.device)
        attn_mask.masked_fill_(~allowed, float("-inf"))
        o_h = F.scaled_dot_product_attention(
            q_h.unsqueeze(0), k_h.unsqueeze(0), v_h.unsqueeze(0),
            attn_mask=attn_mask.unsqueeze(0).unsqueeze(0), scale=scale,
        ).squeeze(0)                                            # [nq, q_len, hd]
        outs.append(o_h.transpose(0, 1).reshape(qe - qs, nq, hd))
    return torch.cat(outs, dim=0)


def paged_decode(q, k_cache, v_cache, block_tables, context_lens, scale):
    """CUDA-graph-safe single-query decode (one query token per seq).

    Fully static-shape / no `.item()` / no python loops, so it can be captured
    by torch.cuda.graph: gather ALL max_blocks per seq, then mask positions
    >= context_lens. q: [bs, nq, hd]; block_tables: [bs, max_blocks];
    context_lens: [bs]. Returns [bs, nq, hd].
    """
    bs, nq, hd = q.shape
    max_blocks = block_tables.shape[1]
    block_size = k_cache.shape[1]
    n_kv = k_cache.shape[2]
    max_ctx = max_blocks * block_size
    rep = nq // n_kv

    bt = block_tables.clamp_min(0).to(torch.long)              # [bs, max_blocks]
    kg = k_cache[bt].reshape(bs, max_ctx, n_kv, hd)            # [bs, ctx, n_kv, hd]
    vg = v_cache[bt].reshape(bs, max_ctx, n_kv, hd)
    # -> [bs, nq, ctx, hd] with GQA expansion
    kg = kg.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)
    vg = vg.permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)
    qh = q.unsqueeze(2)                                        # [bs, nq, 1, hd]

    pos = torch.arange(max_ctx, device=q.device)
    allowed = pos[None, :] < context_lens.to(torch.long)[:, None]   # [bs, ctx]
    attn_mask = torch.zeros(bs, 1, 1, max_ctx, dtype=q.dtype, device=q.device)
    attn_mask.masked_fill_(~allowed[:, None, None, :], float("-inf"))

    o = F.scaled_dot_product_attention(qh, kg, vg, attn_mask=attn_mask, scale=scale)
    return o.squeeze(2).reshape(bs, nq, hd)


def paged_verify(q, k_cache, v_cache, block_tables, context_lens, q_len, scale):
    """CUDA-graph-safe multi-query verify (uniform q_len=K+1 per seq).

    Static-shape variant of the verify path: gather all max_blocks, apply a
    tail-aligned causal mask (query i sits at abs pos context_lens-q_len+i).
    q: [bs*q_len, nq, hd]; block_tables [bs, max_blocks]; context_lens [bs].
    """
    total, nq, hd = q.shape
    bs = total // q_len
    max_blocks = block_tables.shape[1]
    block_size = k_cache.shape[1]
    n_kv = k_cache.shape[2]
    max_ctx = max_blocks * block_size
    rep = nq // n_kv

    bt = block_tables.clamp_min(0).to(torch.long)
    kg = k_cache[bt].reshape(bs, max_ctx, n_kv, hd).permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)
    vg = v_cache[bt].reshape(bs, max_ctx, n_kv, hd).permute(0, 2, 1, 3).repeat_interleave(rep, dim=1)
    qh = q.reshape(bs, q_len, nq, hd).permute(0, 2, 1, 3)      # [bs, nq, q_len, hd]

    pos = torch.arange(max_ctx, device=q.device)              # [ctx]
    qi = torch.arange(q_len, device=q.device)                 # [q_len]
    cl = context_lens.to(torch.long)                          # [bs]
    absq = cl[:, None] - q_len + qi[None, :]                  # [bs, q_len] abs pos of each query
    allowed = pos[None, None, :] <= absq[:, :, None]          # [bs, q_len, ctx] tail-causal
    attn_mask = torch.zeros(bs, 1, q_len, max_ctx, dtype=q.dtype, device=q.device)
    attn_mask.masked_fill_(~allowed[:, None, :, :], float("-inf"))

    o = F.scaled_dot_product_attention(qh, kg, vg, attn_mask=attn_mask, scale=scale)  # [bs,nq,q_len,hd]
    return o.permute(0, 2, 1, 3).reshape(total, nq, hd)


class EagerTreeAttn:
    """Drop-in replacement for flashinfer.BatchPrefillWithPagedKVCacheWrapper
    used by SSD's async tree-decode path (eager only).

    Implements the subset of the flashinfer wrapper API that SSD calls:
      - .plan(qo_indptr, kv_indptr, kv_indices, kv_last_page_len,
              num_qo_heads, num_kv_heads, head_dim, block_size,
              custom_mask=<bool [sum_b MQ_b*ctx_b]>, q_data_type=, kv_data_type=)
      - .run(q, (k_cache, v_cache)) -> [total_q, num_qo_heads, head_dim]

    Attention is done per sequence with torch SDPA under the packed boolean tree
    mask (True = attend). GQA is expanded by repeating KV heads.
    """

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
        gathered = cache[seq_pages]                              # [n_pages, block, n_kv, hd]
        gathered = gathered.reshape(-1, self.num_kv_heads, self.head_dim)
        return gathered[:ctx_len]                                # [ctx_len, n_kv, hd]

    def run(self, q, kv):
        assert self._planned, "EagerTreeAttn.run called before plan"
        k_cache, v_cache = kv
        qo_indptr = self.qo_indptr
        kv_indptr = self.kv_indptr
        bs = qo_indptr.numel() - 1
        scale = self.head_dim ** -0.5
        rep = self.num_qo_heads // self.num_kv_heads

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

            q_seq = q[qs:qe]                                           # [mq, n_qo, hd]
            # -> [n_qo, mq, hd] / [n_qo, ctx, hd]
            q_h = q_seq.transpose(0, 1)
            k_h = k_seq.transpose(0, 1).repeat_interleave(rep, dim=0)
            v_h = v_seq.transpose(0, 1).repeat_interleave(rep, dim=0)

            n = mq * ctx_len
            m = self.custom_mask[mask_off:mask_off + n].view(mq, ctx_len).to(torch.bool)
            mask_off += n
            attn_mask = torch.zeros(mq, ctx_len, dtype=q_h.dtype, device=q_h.device)
            attn_mask.masked_fill_(~m, float("-inf"))

            o_h = F.scaled_dot_product_attention(
                q_h.unsqueeze(0), k_h.unsqueeze(0), v_h.unsqueeze(0),
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(0), scale=scale,
            ).squeeze(0)                                              # [n_qo, mq, hd]
            outs.append(o_h.transpose(0, 1).reshape(mq, self.num_qo_heads, self.head_dim))
        return torch.cat(outs, dim=0)
