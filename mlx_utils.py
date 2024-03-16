import torch
import mlx.core as mx
import mlx.nn as nn

def mlx_graph_for_residual(device='mps', dtype=torch.float16, dim=32000, n_warmups=3, mempool=None):
    static_p = mx.full((dim,), 1, dtype=mx.dtype_from_torch(dtype), device=device)
    static_q = mx.full((dim,), 0, dtype=mx.dtype_from_torch(dtype), device=device)

    @mx.graph(pool=mempool)
    def run(p, q):
        p_mx = mx.from_torch(p)
        q_mx = mx.from_torch(q)
        return mx.maximum(p_mx - q_mx, 0) / mx.sum(mx.maximum(p_mx - q_mx, 0), axis=-1, keepdims=True)

    for _ in range(n_warmups):
        run(static_p.to_torch(), static_q.to_torch())

    return run

def mlx_graph_for_sampling_without_replacement(device='mps', dtype=torch.float16, dim=32000, max_length=384, n_warmups=3, mempool=None, idx_len=8, num_samples=16, temperature=0.6, tree_size=64):
    static_sampling_logits = mx.full((idx_len, dim), 1, dtype=mx.dtype_from_torch(dtype), device=device)
    static_rand = mx.random.uniform(size=(idx_len, dim), dtype=mx.dtype_from_torch(dtype), device=device)

    @mx.graph(pool=mempool)
    def run(draft_logits, rand_vector):
        draft_logits_mx = mx.from_torch(draft_logits)
        rand_vector_mx = mx.from_torch(rand_vector)
        sampling_q = mx.softmax(draft_logits_mx / temperature, axis=-1)
        return mx.topk((rand_vector_mx.log() / sampling_q), k=num_samples, axis=-1).indices.flatten()

    for _ in range(n_warmups):
        run(static_sampling_logits.to_torch(), static_rand.to_torch())

    return run

def mlx_graph_for_sampling_argmax(device='mps', dtype=torch.float16, dim=32000, max_length=384, n_warmups=3, mempool=None, idx_len=8, num_samples=16, temperature=0.6, tree_size=64):
    static_sampling_logits = mx.full((idx_len, dim), 1, dtype=mx.dtype_from_torch(dtype), device=device)

    @mx.graph(pool=mempool)
    def run(draft_logits):
        draft_logits_mx = mx.from_torch(draft_logits)
        return mx.topk(draft_logits_mx, k=num_samples, axis=-1).indices.flatten()

    for _ in range(n_warmups):
        run(static_sampling_logits.to_torch())

    return run

def mlx_graph_for_sampling_with_replacement(device='mps', dtype=torch.float16, dim=32000, max_length=384, n_warmups=3, mempool=None, idx_len=8, num_samples=16, temperature=0.6, tree_size=64):
    static_sampling_logits = mx.full((idx_len, dim), 1, dtype=mx.dtype_from_torch(dtype), device=device)

    @mx.graph(pool=mempool)
    def run(draft_logits):
        draft_logits_mx = mx.from_torch(draft_logits)
        sampling_q = mx.softmax(draft_logits_mx / temperature, axis=-1)
        return mx.random.categorical(sampling_q, num_samples=num_samples, dtype=mx.int64).flatten()

    for _ in range(n_warmups):
        run(static_sampling_logits.to_torch())

    return run

def get_residual(p: torch.Tensor, q: torch.Tensor):
    p_mx = mx.from_torch(p)
    q_mx = mx.from_torch(q)
    residual = mx.maximum(p_mx - q_mx, 0)
    residual = residual / mx.sum(residual, axis=-1, keepdims=True)
    return residual.to_torch()

def sampling_without_replacement(sampling_logits: torch.Tensor, rand: torch.Tensor, num_samples: int, temperature: float):
    sampling_logits_mx = mx.from_torch(sampling_logits)
    rand_mx = mx.from_torch(rand)
    sampling_q = mx.softmax(sampling_logits_mx / temperature, axis=-1)
    position = mx.topk((rand_mx.log() / sampling_q), k=num_samples, axis=-1).indices.flatten()
    return position.to_torch()

def sampling_with_replacement(sampling_logits: torch.Tensor, num_samples: int, temperature: float):
    sampling_logits_mx = mx.from_torch(sampling_logits)
    sampling_q = mx.softmax(sampling_logits_mx / temperature, axis=-1)
    position = mx.random.categorical(sampling_q, num_samples=num_samples, dtype=mx.int64).flatten()
    return position.to_torch()

def sampling_argmax(sampling_logits: torch.Tensor, num_samples: int):
    sampling_logits_mx = mx.from_torch(sampling_logits)
    return mx.topk(sampling_logits_mx, k=num_samples, axis=-1).indices.flatten().to_torch()

def make_tree_attention_mask(prefix_len: int, gen_len: int, ancestors: list[list[int]], device='mps', dtype=torch.float32) -> torch.Tensor:
    tree_mask = mx.full((gen_len, gen_len + prefix_len), float('-inf'), dtype=mx.dtype_from_torch(dtype), device=device)
    for idx, ancestor in enumerate(ancestors):
        if len(ancestor) > 0:
            tree_mask[idx][ancestor] = 0.0
    return tree_mask[None, None, :, :].to_torch()

