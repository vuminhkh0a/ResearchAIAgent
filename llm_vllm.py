"""vLLM-based LLM factory with Qwen3-4B-Int4, KV-cache, and optimized settings.

Optimizations for RTX 3060/4060 (8-12GB VRAM):
- AWQ 4-bit quantization (Qwen3-4B-Instruct-AWQ)
- Prefix caching (KV-cache reuse across requests)
- Flash attention (enabled by default in vLLM 0.6+)
- Continuous batching (max_num_batched_tokens)
- Optimized sampling params for speed (top_k=1 for greedy-ish)
- torch.compile equivalent via enforce_eager=False + CUDA graphs
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.callbacks import CallbackManagerForLLMRun
from pydantic import Field

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.entrypoints.chat_utils import ChatCompletionMessageParam

from config.settings import settings


class VLLMChatModel(BaseChatModel):
    """LangChain-compatible wrapper for vLLM with optimized settings for speed."""
    
    model_name: str = Field(default="Qwen/Qwen3-4B-Instruct")
    tensor_parallel_size: int = Field(default=1)
    gpu_memory_utilization: float = Field(default=0.85)
    max_model_len: int = Field(default=8192)
    dtype: str = Field(default="half")
    quantization: str = Field(default="awq")
    enforce_eager: bool = Field(default=False)  # False = CUDA graphs (like torch.compile)
    enable_prefix_caching: bool = Field(default=True)  # KV-cache reuse
    max_num_batched_tokens: int = Field(default=4096)
    disable_custom_all_reduce: bool = Field(default=True)  # Faster for single GPU
    
    _llm: "LLM" = None
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._llm = self._create_llm()
    
    def _create_llm(self) -> "LLM":
        from vllm import LLM
        
        # Windows compatibility: detect CUDA and fallback to CPU
        use_cuda = False
        try:
            import torch
            use_cuda = torch.cuda.is_available()
        except Exception:
            pass
        
        if not use_cuda:
            print("CUDA not available, using CPU mode (slower but functional)")
            return LLM(
                model=self.model_name,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.0,
                max_model_len=self.max_model_len,
                dtype=self.dtype,
                quantization=self.quantization,
                enforce_eager=True,  # CPU mode needs eager
                enable_prefix_caching=self.enable_prefix_caching,
                max_num_batched_tokens=self.max_num_batched_tokens,
                device="cpu",
                trust_remote_code=True,
            )
        
        return LLM(
            model=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            quantization=self.quantization,
            enforce_eager=self.enforce_eager,
            enable_prefix_caching=self.enable_prefix_caching,
            max_num_batched_tokens=self.max_num_batched_tokens,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
            trust_remote_code=True,
            swap_space=0,
            block_size=16,
        )
    
    @property
    def _llm_type(self) -> str:
        return "vllm-qwen3"
    
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs,
    ) -> ChatResult:
        from vllm.entrypoints.chat_utils import load_chat_template
        from vllm import SamplingParams
        
        # Convert LangChain messages to vLLM chat format
        chat_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": msg.content or ""})
        
        # Load Qwen3 chat template (cached)
        chat_template = load_chat_template(self._llm.get_tokenizer(), chat_template="qwen")
        
        # Optimized sampling for speed: near-greedy with slight randomness
        temperature = kwargs.get("temperature", settings.llm_temperature)
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.95 if temperature > 0 else 1.0,
            top_k=1 if temperature == 0 else 20,  # Greedy when temp=0
            max_tokens=kwargs.get("max_tokens", settings.reserved_response_tokens),
            stop=stop,
            skip_special_tokens=True,
            # Speed optimizations
            use_beam_search=False,
            length_penalty=1.0,
        )
        
        # Use vLLM's chat completion
        outputs = self._llm.chat(
            messages=[chat_messages],
            sampling_params=sampling_params,
            chat_template=chat_template,
            use_tqdm=False,
        )
        
        content = outputs[0].outputs[0].text
        message = AIMessage(content=content)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
    
    def bind_tools(self, tools):
        """Bind tools for function calling - returns self with tool info attached."""
        bound = self.__class__(
            model_name=self.model_name,
            tensor_parallel_size=self.tensor_parallel_size,
            gpu_memory_utilization=self.gpu_memory_utilization,
            max_model_len=self.max_model_len,
            dtype=self.dtype,
            quantization=self.quantization,
            enforce_eager=self.enforce_eager,
            enable_prefix_caching=self.enable_prefix_caching,
            max_num_batched_tokens=self.max_num_batched_tokens,
            disable_custom_all_reduce=self.disable_custom_all_reduce,
        )
        bound._bound_tools = tools
        return bound
    
    def invoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """Synchronous invoke for compatibility."""
        result = self._generate(messages, **kwargs)
        return result.generations[0].message
    
    async def ainvoke(self, messages: list[BaseMessage], **kwargs) -> AIMessage:
        """Async invoke - delegates to sync for now."""
        return self.invoke(messages, **kwargs)
    
    def stream(self, messages: list[BaseMessage], **kwargs):
        """Stream tokens for perceived latency improvement."""
        from vllm import SamplingParams
        from vllm.entrypoints.chat_utils import load_chat_template
        
        chat_messages: list[ChatCompletionMessageParam] = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                chat_messages.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                chat_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                chat_messages.append({"role": "assistant", "content": msg.content or ""})
        
        chat_template = load_chat_template(self._llm.get_tokenizer(), chat_template="qwen")
        
        temperature = kwargs.get("temperature", settings.llm_temperature)
        sampling_params = SamplingParams(
            temperature=temperature if temperature > 0 else 0.0,
            top_p=0.95 if temperature > 0 else 1.0,
            top_k=1 if temperature == 0 else 20,
            max_tokens=kwargs.get("max_tokens", settings.reserved_response_tokens),
            skip_special_tokens=True,
        )
        
        # Stream using vLLM's generator
        for output in self._llm.chat(
            messages=[chat_messages],
            sampling_params=sampling_params,
            chat_template=chat_template,
            use_tqdm=False,
        ):
            for token_output in output.outputs:
                if token_output.text:
                    yield AIMessage(content=token_output.text)


@lru_cache(maxsize=1)
def get_vllm_model() -> VLLMChatModel:
    """Get singleton vLLM model instance."""
    return VLLMChatModel()


def get_llm(*, temperature: float | None = None) -> VLLMChatModel:
    """Factory function compatible with existing code."""
    model = get_vllm_model()
    if temperature is not None:
        # Create a new instance with different temperature if needed
        return VLLMChatModel()
    return model