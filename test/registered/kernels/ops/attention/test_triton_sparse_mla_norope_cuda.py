# SPDX-License-Identifier: Apache-2.0
"""Numerical coverage for GLM-5.3 NoPE sparse MLA on CUDA."""

import pytest
import torch

from sglang.kernels.ops.attention.dsa.triton_sparse_mla_decode import (
    triton_sparse_mla_decode_splitk,
)
from sglang.kernels.ops.attention.dsa.triton_sparse_mla_prefill import (
    sparse_mla_prefill,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-large")


@pytest.mark.parametrize("path", ["prefill", "decode"])
def test_norope_dv256_kpool_tail_matches_torch(path):
    torch.manual_seed(0)
    tokens, heads, value_dim, topk = 1, 16, 256, 2051
    scale = value_dim**-0.5
    q = torch.randn((tokens, heads, value_dim), device="cuda", dtype=torch.bfloat16)
    q_rope = torch.empty((tokens, heads, 0), device="cuda", dtype=torch.bfloat16)
    kv = torch.randn((topk, 1, value_dim), device="cuda", dtype=torch.bfloat16)
    indices = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(
        tokens, 1, topk
    )

    if path == "prefill":
        actual = sparse_mla_prefill(
            q=q,
            kv=kv,
            indices=indices,
            sm_scale=scale,
            d_v=value_dim,
        )
    else:
        actual = triton_sparse_mla_decode_splitk(
            q_nope=q,
            q_rope=q_rope,
            kv=kv,
            indices=indices,
            sm_scale=scale,
            d_v=value_dim,
        ).squeeze(0)

    expected = (
        torch.softmax(
            torch.einsum("thd,kd->thk", q.float(), kv[:, 0].float()) * scale,
            dim=-1,
        )
        @ kv[:, 0].float()
    )
    torch.cuda.synchronize()
    error = (actual.float() - expected).abs()
    assert tuple(actual.shape) == (tokens, heads, value_dim)
    assert error.max().item() < 0.03
    assert error.mean().item() < 0.003
