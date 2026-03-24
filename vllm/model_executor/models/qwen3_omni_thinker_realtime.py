# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The Qwen team.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Realtime support for Qwen3-Omni-Thinker models."""

import asyncio
import torch
import collections.abc import AsyncGenerator

import numpy as np

from vllm.config import ModelConfig, VllmConfig
from vllm.inputs.data import PromptType, TokensPrompt
from vllm.model_executor.models.interfaces import SupportsRealtime
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    ISO639_1_SUPPORTED_LANGS,
    Qwen3OmniMoeThinkerForConditionalGeneration,
)
from vllm.model_executor.models.qwen3_omni_moe_thinker import (
    Qwen3OmniMoeThinkerProcessingInfo,
    Qwen3OmniMoeThinkerDummyInputsBuilder,
    Qwen3OmniMoeThinkerMultiModalProcessor,
    _get_feat_extract_output_lengths,
)
from vllm.tokenizers import cached_tokenizer_from_config

logger = init_logger(__name__)

_PRE_ALLOCATE_BUFFER_SIZE_IN_S = 60

class Qwen3OmniRealtimeBuffer:
    """Audio buffer for Qwen3-Omni realtime streaming.

    Accumulates audio samples and yields segments when enough
    audio has been buffered for processing.
    """

    def __init__(self, sampling_rate: int, segment_duration_s: float = 3.0):
        self._sampling_rate = sampling_rate
        self._segment_size = int(segment_duration_s * sampling_rate)

        self._buffer_size = _PRE_ALLOCATE_BUFFER_SIZE_IN_S * sampling_rate
        self._buffer: np.ndarry = np.empty(self._buffer_size, dtype=np.float32)
        self._filled_len = 0

    def write_audio(self, audio: np.ndarray) -> None:
        put_end = self._filled_len + len(audio)
        if put_end > self._buffer_size:
            new_size = max(self._buffer_size * 2, put_end)
            new_buffer = np.empty(new_size, dtype=np.float32)
            new_buffer[: self._filled_len] = self._buffer[: self._filled_len]
            self._buffer = new_buffer
            self._buffer_size = new_size

        self._buffer[self._filled_len : put_end] = audio
        self._filled_len = put_end
  
