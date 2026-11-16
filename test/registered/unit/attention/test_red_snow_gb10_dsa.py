# SPDX-License-Identifier: Apache-2.0
"""Admission contracts for GLM-5.3 NoPE DSA on GB10."""

from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import _check_dsa_backend_constraints
from sglang.srt.layers.attention.dsa.dsa_backend_kpool import (
    DeepseekSparseAttnBackendKPoolMixin,
)
from sglang.srt.layers.attention.dsa_backend import (
    _validate_triton_sparse_mla_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


def test_cuda_admits_triton_decode_with_sparse_mla_prefill():
    with patch(
        "sglang.srt.arg_groups.overrides.get_platform",
        return_value=SimpleNamespace(is_sm120=True),
    ):
        _check_dsa_backend_constraints(
            "bfloat16",
            "triton_sparse_mla",
            "triton",
            hip=False,
        )


def test_sparse_mla_prefill_admits_glm_kpool_tail_width():
    _validate_triton_sparse_mla_backend(
        device_sm_major=12,
        num_q_heads=16,
        union=0,
        index_kpool=4,
    )


def test_kpool_tail_guard_admits_triton_backends_on_bf16_pool():
    backend = SimpleNamespace(
        dsa_index_kpool=4,
        device_sm_major=12,
        dsa_kv_cache_store_fp8=False,
    )
    indices = torch.full((1, 2051), -1, dtype=torch.int32)
    for implementation, phase in (
        ("triton_sparse_mla", "prefill"),
        ("triton", "decode"),
    ):
        DeepseekSparseAttnBackendKPoolMixin._check_kpool_tail_backend(
            backend,
            indices,
            implementation,
            phase,
        )
