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

    def read_audio(self) -> np.ndarray | None:
        if self._filled_len < self._segment_size:
            return None

        segment = self._buffer[: self._segment_size].copy()
        remaining = self._filled_len - self._segment_size
        if remaining > 0:
            self._buffer[: remaining] = self._buffer[self._segment_size : self._filled_len]
        self._filled_len = remaining
        return segment

    def flush(self) -> np.ndarray | None:
        if self._filled_len == 0:
            return None
        audio = self._buffer[: self._filled_len].copy()
        self._filled_len = 0
        return audio


class Qwen3OmniRealtimeMultiModalProcessor(Qwen3OmniMoeThinkerMultiModalProcessor):
    """Multi-modal processor for Qwen3-Omni realtime audio processing."""

    def __init__(
        self,
        info: _I,
        dummy_inputs: BaseDummyInputsBuilder[_I],
        *,
        cache: BaseMultiModalProcessorCache | None = None,
    ) -> None:
        self._info = info
        self._feature_extractor = info.get_feature_extractor()
        self._spatial_merge_size = info.get_hf_config().vision_config.spatial_merge_size
        super().__init__(info, dummy_inputs, cache=None)

    def _maybe_apply_prompt_updates(
        self,
        mm_items: MultiModalDataItems,
        prompt_ids: list[int],
        mm_kwargs: MultiModalKwargsOptionalItems,
        mm_prompt_updates: MultiModalPromptUpdates,
        is_update_applied: bool,
    ) -> tuple[list[int], dict[str, list[PlaceholderFeaturesInfo]]]:
        audios = mm_kwargs.get("audio", [])
        if len(audios) == 0:
            prompt_ids, mm_placeholders = self._apply_prompt_updates(
                prompt_ids,
                mm_prompt_updates,
            )
            return prompt_ids, mm_placeholders
        assert len(audios) == 1, (
            f"Expected only one audio input for realtime, got {len(audios)}"
        )

        audio_data = audios[0]

        # Calculate audio feature length from raw audio samples
        raw_audio = audio_data.get("input_audio_features")
        audio_feature_lengths = audio_data.get("audio_feature_lengths")
        if audio_feature_lengths is not None:
            if isinstance(audio_feature_lengths.data, torch.Tensor):
                audio_len = _get_feat_extract_output_lengths(
                    audio_feature_lengths.data
                ).item()
            else:
                audio_len = int(
                    _get_feat_extract_output_lengths(
                        torch.tensor(audio_feature_lengths.data)
                    ).item()
                )
        else:
            audio_len = 0

        # Get audio_pad token ID and expand placeholder in prompt_ids
        # so that MRoPE position computation matches seq_len.
        tokenizer = self.info.get_tokenizer()
        audio_pad_id = tokenizer.convert_tokens_to_ids("<|audio_pad|>")

        # Find the audio_pad token position and expand it to audio_len tokens
        expanded_ids = list[int]()
        pad_start_idx = -1
        for i, tid in enumerate(prompt_ids):
            if tid == audio_pad_id and pad_start_idx == -1:
                pad_start_idx = i
                expanded_ids.extend([audio_pad_id] * audio_len)
            else:
                expanded_ids.append(tid)

        if pad_start_idx == -1:
            pad_start_idx = 0

        features_info = PlaceholderFeaturesInfo(
            modality="audio",
            item_idx=0,
            start_idx=pad_start_idx,
            tokens=audio_len * [audio_pad_id],
            is_embed=None,
        )
        return expanded_ids, {"audio": [features_info]}


# NOTE: A separate model class is required here because the multimodal
# processor registry binds one processor per model class. The realtime
# endpoint needs a different processor (Qwen3OmniRealtimeMultiModalProcessor)
# than the base transcription endpoint, so we register it on this subclass.
@MULTIMODAL_REGISTRY.register_processor(
    Qwen3OmniRealtimeMultiModalProcessor,
    info=Qwen3OmniMoeThinkerProcessingInfo,
    dummy_inputs=Qwen3OmniMoeThinkerDummyInputsBuilder,
)
class Qwen3OmniRealtimeGeneration(
    Qwen3OmniMoeThinkerForConditionalGeneration,
    SupportsRealtime,
):
    """Qwen3-Omni-Thinker with Realtime audio streaming support.

    This class extends Qwen3OmniMoeThinkerForConditionalGeneration to support
    low-latency streaming audio-to-text transcription via WebSocket.

    The key difference from the base class is the realtime processor which
    handles audio buffering, segmentation, and streaming output.
    """

    realtime_max_tokens = 1
    supported_languages = ISO639_1_SUPPORTED_LANGS

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

    @classmethod
    async def buffer_realtime_audio(
        cls,
        audio_stream: AsyncGenerator[np.ndarry, None],
        input_stream: asyncio.Queue[list[int]],
        model_config: ModelConfig,
    ) -> AsyncGenerator[PromptType, None]:
        """
        Buffer and process streaming audio for realtime transcription.

        This method:
        1. Receives audio chunks from the WebSocket client
        2. Accumulates audio in a buffer
        3. Yields segments when enough audio is buffered
        4. Converts audio to format expected by Qwen3-Omni-Thinker

        Args:
            audio_stream: Async generator yielding float32 numpy arrays
            input_stream: Queue containing context token IDs from previous outputs
            model_config: Model configuration

        Yields:
            TokensPrompt objects containing audio placeholders and raw audio data
        """
        from vllm.transformers_utils.processor import cached_processor_from_config

        processor = cached_processor_from_config(model_config)
        feature_extractor = processor.feature_extractor
        sampling_rate = feature_extractor.sampling_rate
        tokenizer = cached_tokenizer_from_config(model_config)

        # Use a small segment size for low-latency streaming
        segment_duration_s = 4
        buffer = Qwen3OmniRealtimeBuffer(
            sampling_rate=sampling_rate,
            segment_duration_s=segment_duration_s,
        )

        # Get audio placeholder string
        audio_placeholder = cls.get_placeholder_str("audio", 0)

        # Create prompt template for transcription
        # Using the format expected by Qwen3-Omni-Thinker
        prompt_template1 = f"<|im_start|>user\\n<|audio_start|><|audio_pad|>"
        prompt_template2 = f"<|audio_pad|>"
        prompt_template3 = f"<|audio_pad|><|audio_end|><|im_end|>\\n<|im_start|>assistant\\n"
        prompt_token_ids1 = tokenizer.encode(prompt_template1)
        prompt_token_ids2 = tokenizer.encode(prompt_template2)
        prompt_token_ids3 = tokenizer.encode(prompt_template3)

        prompt_template = f"<|im_start|>user\\n{audio_placeholder}<|im_end|>\\n<|im_start|>assistant\\n"
        prompt_token_ids = tokenizer.encode(prompt_template)

        frist = True
        async for aduio_chunk in audio_stream:
            if audio_chunk.dtype == np.int16:
                audio_chunk = audio_chunk.astype(np.float32) / 32768.0

            if len(audio_chunk.shape) > 1:
                if audio_chunk.shape[1] == 1:
                    audio_chunk = audio_chunk.squeeze(-1)
                else:
                    audio_chunk = np.mean(audio_chunk, axis=-1)

            buffer.write_audio(audio_chunk)

            # Process segments as they become available
            while (segment := buffer.read_audio()) is not None:
                if frist:
                    prompt_token_ids = prompt_tokens_ids1
                    frist = False
                else:
                    prompt_token_ids = prompt_tokens_ids2
                yield TokensPrompt(
                    prompt_token_ids=pront_token_ids,
                    multi_modal_data={"audio": segment},
                ), False

        if not frist:
            prompt_token_ids = prompt_token_ids3
        remaining = buffer.flush()
        if remaining is not None and len(remaining) > 0:
            yield TokensPrompt(
                prompt_token_ids=prompt_token_ids,
                multi_modal_data={"audio": remaining},
            ), True


__all__ = [
    "Qwen3OmniRealtimeGeneration",
    "Qwen3OmniRealtimeBuffer",
    "Qwen3OmniRealtimeMultiModalProcessor",
]


if __name__ == "__main__":
    print("Qwen3OmniRealtimeGeneration module loaded successfully")
    print(f"Supports Realtime: {Qwen3OmniRealtimeGeneration.supports_realtime}")
    print(f"Realtime max tokens: {Qwen3OmniRealtimeGeneration.realtime_max_tokens}")
        
