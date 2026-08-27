#!/usr/bin/env python
# -*- coding: utf-8 -*-
#  Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.

"""Vocab embedding padding patch for heterogeneous TP.

Logits/embeddings keep uniform partition sizes (their all-gather needs equal
tensor shapes on every rank), so the vocab is padded to
``lcm(padding_size, local_tp_size)`` instead of using asymmetric sharding.
"""

from __future__ import annotations

from math import lcm

_PATCHED = False

DEFAULT_PADDING_SIZE = 64


def _is_hetero_tp() -> bool:
    from vllm.config import get_current_vllm_config_or_none

    cfg = get_current_vllm_config_or_none()
    return bool(
        cfg is not None
        and getattr(cfg.parallel_config, "is_heterogeneous_tp", False)
    )


def _patched_vllm_vocab_init(
    self,
    num_embeddings,
    embedding_dim,
    params_dtype=None,
    org_num_embeddings=None,
    padding_size=DEFAULT_PADDING_SIZE,
    *args,
    **kwargs,
):
    if padding_size is None:
        padding_size = DEFAULT_PADDING_SIZE
    if _is_hetero_tp():
        from vllm.distributed import get_tensor_model_parallel_world_size

        padding_size = lcm(
            padding_size, get_tensor_model_parallel_world_size()
        )
    return _ORIG_VLLM_VOCAB_INIT(
        self,
        num_embeddings,
        embedding_dim,
        params_dtype=params_dtype,
        org_num_embeddings=org_num_embeddings,
        padding_size=padding_size,
        *args,
        **kwargs,
    )


def _patched_ascend_vocab_init(
    self,
    num_embeddings,
    embedding_dim,
    params_dtype=None,
    org_num_embeddings=None,
    padding_size=DEFAULT_PADDING_SIZE,
    *args,
    **kwargs,
):
    if padding_size is None:
        padding_size = DEFAULT_PADDING_SIZE
    if _is_hetero_tp():
        from vllm.distributed import get_tensor_model_parallel_world_size

        padding_size = lcm(
            padding_size, get_tensor_model_parallel_world_size()
        )
    return _ORIG_ASCEND_VOCAB_INIT(
        self,
        num_embeddings,
        embedding_dim,
        params_dtype=params_dtype,
        org_num_embeddings=org_num_embeddings,
        padding_size=padding_size,
        *args,
        **kwargs,
    )


_ORIG_VLLM_VOCAB_INIT = None
_ORIG_ASCEND_VOCAB_INIT = None


def apply_hetero_vocab_patch():
    global _PATCHED, _ORIG_VLLM_VOCAB_INIT, _ORIG_ASCEND_VOCAB_INIT
    if _PATCHED:
        return

    import vllm.model_executor.layers.vocab_parallel_embedding as vllm_vpe

    _ORIG_VLLM_VOCAB_INIT = vllm_vpe.VocabParallelEmbedding.__init__
    vllm_vpe.VocabParallelEmbedding.__init__ = _patched_vllm_vocab_init

    try:
        import vllm_ascend.ops.vocab_parallel_embedding as ascend_vpe

        _ORIG_ASCEND_VOCAB_INIT = (
            ascend_vpe.AscendVocabParallelEmbedding.__init__
        )
        ascend_vpe.AscendVocabParallelEmbedding.__init__ = (
            _patched_ascend_vocab_init
        )
    except Exception:
        _ORIG_ASCEND_VOCAB_INIT = None

    _PATCHED = True
