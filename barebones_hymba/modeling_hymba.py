import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)

from transformers.modeling_outputs import (
    MoeCausalLMOutputWithPast,
    MoeModelOutputWithPast,
    SequenceClassifierOutputWithPast,
)

from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

from transformers.utils.import_utils import is_torch_fx_available
from configuration_hymba import HymbaConfig
from torch.utils.checkpoint import checkpoint

# Optional imports - handle gracefully if not available
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False
    _flash_supports_window_size = False
    print("Warning: flash_attn not available. Some features may be limited.")

try:
    from einops import rearrange, repeat, reduce, pack, unpack
    from einops.layers.torch import Rearrange
    EINOPS_AVAILABLE = True
except ImportError:
    EINOPS_AVAILABLE = False
    print("Warning: einops not available. Some features may be limited.")

if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx
    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)

try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    
    is_fast_path_available = all(
        (selective_state_update, selective_scan_fn, causal_conv1d_fn, causal_conv1d_update, mamba_inner_fn)
    )
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    is_fast_path_available = False
    print("Warning: mamba_ssm not available. Mamba functionality will be limited.")

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "HymbaConfig"


def pad_at_dim(t, pad: Tuple[int, int], dim = -1, value = 0.):
    """Pad tensor at specific dimension"""
    if pad == (0, 0):
        return t

    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value = value)


def load_balancing_loss_func(
        gate_logits: torch.Tensor, num_experts: torch.Tensor = None, top_k=2, attention_mask: Optional[torch.Tensor] = None
) -> float:
    """
    Computes auxiliary load balancing loss as in Switch Transformer
    """
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat(
            [layer_gate.to(compute_device) for layer_gate in gate_logits if layer_gate.shape[1] > 1], dim=0
        )

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
                .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
                .reshape(-1, top_k, num_experts)
                .to(compute_device)
        )

        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )

        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
                .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
                .reshape(-1, num_experts)
                .to(compute_device)
        )

        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts


def _get_unpad_data(attention_mask):
    """Get unpadding data for flash attention"""
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class HymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        HymbaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class PerheadHymbaRMSNorm(nn.Module):
    def __init__(self, hidden_size, num_heads, eps=1e-6):
        """
        For per-head kq normalization
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, num_heads, 1, hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        assert hidden_states.shape[1] == self.weight.shape[1], f"hidden_state: {hidden_states.shape}, weight: {self.weight.shape}"
        assert hidden_states.shape[3] == self.weight.shape[3], f"hidden_state: {hidden_states.shape}, weight: {self.weight.shape}"
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class HymbaOnlyNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Normalization without learnable parameters
        """
        super().__init__()
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return hidden_states.to(input_dtype)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config, dim, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.base = base
        self.config = config

        self.rope_type = config.rope_type
        self.factor = 2

        max_position_embeddings = self.config.max_position_embeddings

        if config.rope_type is None or config.rope_type == "default":
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.max_seq_len_cached = max_position_embeddings

        elif config.rope_type == 'ntk':
            assert self.config.orig_max_position_embeddings is not None
            orig_max_position_embeddings = self.config.orig_max_position_embeddings

            base = base * ((self.factor * max_position_embeddings / orig_max_position_embeddings) - (self.factor - 1)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

            self.max_seq_len_cached = orig_max_position_embeddings

        elif config.rope_type == 'dynamic_ntk':
            inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
            self.original_inv_freq = inv_freq
            self.max_seq_len_cached = self.config.orig_max_position_embeddings

        else:
            raise ValueError(f"Not support rope_type: {config.rope_type}")

        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _dynamic_frequency_update(self, position_ids, device):
        """Dynamic RoPE frequency update"""
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            base = self.base * ((self.factor * seq_len / self.config.orig_max_position_embeddings) - (self.factor - 1)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))

            self.register_buffer("inv_freq", inv_freq, persistent=False)
            self.max_seq_len_cached = seq_len

        if seq_len < self.config.orig_max_position_embeddings and self.max_seq_len_cached > self.config.orig_max_position_embeddings:  # reset
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.config.orig_max_position_embeddings

    @torch.no_grad()
    def forward(self, x, position_ids):
        if self.rope_type == 'dynamic_ntk':
            self._dynamic_frequency_update(position_ids, device=x.device)

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)
    else:
        q_embed = None

    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class HybridMambaAttentionDynamicCache(DynamicCache):
    """
    A dynamic cache that can handle both the attention cache (which has a seq_len dimension) and the mamba cache
    (which has a constant shape regardless of seq_len).

    This cache has two sets of lists of tensors: `key_cache` and `value_cache` for attention cache and `conv_states`
    and `ssm_states` for mamba cache. Each of these lists has `num_layers` tensors.
    """

    def __init__(self, config, batch_size, dtype=torch.float16, device=None, layer_type=None):
        self.dtype = dtype
        self.has_previous_state = False  # only used by mamba
        intermediate_size = config.mamba_expand * config.hidden_size
        ssm_state_size = config.mamba_d_state
        conv_kernel_size = config.mamba_d_conv
        self.conv_states = []
        self.ssm_states = []

        self.layer_type = layer_type

        for i in range(config.num_hidden_layers):
            if layer_type is None:
                has_mamba_state = True
            else:
                has_mamba_state = self.layer_type[i] == 'h' or self.layer_type[i] == 'm'

            if has_mamba_state:
                if hasattr(config, 'conv_dim'):
                    conv_dim = config.conv_dim[str(i)]
                else:
                    conv_dim = intermediate_size
                self.conv_states += [
                    torch.zeros(batch_size, conv_dim, conv_kernel_size, device=device, dtype=dtype)
                ]
                self.ssm_states += [
                    torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

        self.mamba_past_length = [0 for _ in range(config.num_hidden_layers)]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Update the cache
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        """Reorders the cache for beam search, given the selected beam indices."""
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states."""
        if self.layer_type[layer_idx] == 'm':
            return self.mamba_past_length[layer_idx]

        if self.key_cache[layer_idx].shape[-1] == 0:
            return 0

        return self.key_cache[layer_idx].shape[-2]

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridMambaAttentionDynamicCache does not have a legacy cache equivalent.")


@dataclass
class MambaCacheParams:
    seqlen_offset: int = 0
    conv_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    ssm_states: Dict[int, torch.Tensor] = field(default_factory=dict)


class HymbaAttention(nn.Module):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention.
    """

    def __init__(self, config: HymbaConfig, layer_idx: Optional[int] = None, reuse_kv=False, output_hidden_size=None, attn_only_wo_proj=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing a `layer_idx` is not recommended and will "
                "lead to errors during the forward call if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.hidden_size = config.attn_hidden_size if config.attn_hidden_size > 0 else config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        self.attn_only_wo_proj = attn_only_wo_proj

        self.kq_head_dim = config.kq_head_dim if config.kq_head_dim > 0 else self.head_dim
        self.v_head_dim = config.v_head_dim if config.v_head_dim > 0 else self.head_dim

        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.is_causal = True
        self.attention_dropout = config.attention_dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        if not self.attn_only_wo_proj:
            self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.kq_head_dim, bias=False)

        self.reuse_kv = reuse_kv

        if not self.attn_only_wo_proj and not self.reuse_kv:
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.kq_head_dim, bias=False)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.v_head_dim, bias=False)

        if output_hidden_size is None:
            output_hidden_size = self.hidden_size

        if not self.attn_only_wo_proj:
            self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, output_hidden_size, bias=False)

        if self.config.kq_norm == "rms":
            self.k_norm = HymbaRMSNorm(self.kq_head_dim)
            self.q_norm = HymbaRMSNorm(self.kq_head_dim)
        elif self.config.kq_norm == "perhead-rms":
            self.k_norm = PerheadHymbaRMSNorm(self.kq_head_dim, self.num_key_value_heads)
            self.q_norm = PerheadHymbaRMSNorm(self.kq_head_dim, self.num_heads)
        elif self.config.kq_norm == "none":
            self.k_norm = None
            self.q_norm = None
        else:
            raise NotImplementedError(f"Unknown kq_norm: {self.config.kq_norm}")

        if self.config.rope:
            self._init_rope()

    def set_rope(self, rope_type, orig_max_position_embeddings, max_position_embeddings):
        self.config.rope_type = rope_type
        self.config.orig_max_position_embeddings = orig_max_position_embeddings
        self.config.max_position_embeddings = max_position_embeddings
        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(
            config=self.config,
            dim=self.kq_head_dim,
            base=self.rope_theta,
            device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            kv_last_layer = None,
            use_swa=False,
            query_states = None,
            key_states=None,
            value_states=None,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        raise NotImplementedError("HymbaAttention is an abstract class. Use one of the subclasses.")


class HymbaPreTrainedModel(PreTrainedModel):
    config_class = HymbaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["HymbaDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class HymbaForCausalLM(HymbaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: HymbaConfig):
        super().__init__(config)
        self.config = config
        # For now, use a simplified model structure
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

        # Simplified model for now - we'll integrate with barebones later
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.final_layernorm = HymbaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            output_router_logits: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            calc_logits_for_entire_prompt: Optional[bool] = True,
    ):
        """
        Simplified forward pass - will be enhanced with full model integration
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Simple embedding for now
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Apply final layernorm
        hidden_states = self.final_layernorm(inputs_embeds)

        # Calculate logits
        if calc_logits_for_entire_prompt:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -1:, :])
        logits = logits.float()

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,)
            return (loss,) + output if loss is not None else output

        from transformers.modeling_outputs import CausalLMOutputWithPast
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
            hidden_states=hidden_states if output_hidden_states else None,
            attentions=None,
        )

    def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            output_router_logits=False,
            **kwargs,
    ):
        """Prepare inputs for generation"""
        # Basic implementation for now
        model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "output_router_logits": output_router_logits,
                "calc_logits_for_entire_prompt": self.config.calc_logits_for_entire_prompt,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
