"""
Cache Management System for Hymba Model
Implements KV cache with cross-layer sharing and sliding window optimization
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
import warnings


class HymbaCache:
    """
    Advanced KV cache management for Hymba model with cross-layer sharing
    and sliding window optimization
    """
    
    def __init__(self, config):
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.sliding_window = getattr(config, 'sliding_window', 1024)
        self.num_memory_tokens = getattr(config, 'num_memory_tokens', 128)
        
        # Cross-layer KV sharing configuration
        self.kv_reuse_group = getattr(config, 'kv_reuse_group', [])
        self.use_cache = getattr(config, 'use_cache', True)
        
        # Cache storage
        self.key_cache: Dict[int, torch.Tensor] = {}
        self.value_cache: Dict[int, torch.Tensor] = {}
        self.cache_position = 0
        self.max_cache_len = getattr(config, 'max_position_embeddings', 8192)
        
        # Cross-layer sharing mappings
        self._build_sharing_mappings()
        
    def _build_sharing_mappings(self):
        """Build mappings for cross-layer KV sharing"""
        self.layer_to_group = {}
        self.group_to_layers = {}
        
        for group_idx, layer_group in enumerate(self.kv_reuse_group):
            self.group_to_layers[group_idx] = layer_group
            for layer_idx in layer_group:
                self.layer_to_group[layer_idx] = group_idx
                
    def get_cache_key(self, layer_idx: int) -> int:
        """Get the cache key for a layer (considering cross-layer sharing)"""
        if layer_idx in self.layer_to_group:
            # Use the first layer in the group as cache key
            group_idx = self.layer_to_group[layer_idx]
            return min(self.group_to_layers[group_idx])
        return layer_idx
        
    def update_cache(
        self, 
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_position: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer
        
        Args:
            layer_idx: Layer index
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            cache_position: Position to update cache at
            
        Returns:
            Updated key_states and value_states including cached values
        """
        if not self.use_cache:
            return key_states, value_states
            
        cache_key = self.get_cache_key(layer_idx)
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Initialize cache if not exists
        if cache_key not in self.key_cache:
            self._initialize_cache(cache_key, batch_size, num_heads, head_dim, key_states.device, key_states.dtype)
            
        # Update cache position
        if cache_position is None:
            cache_position = self.cache_position
            
        # Handle sliding window for long sequences
        effective_cache_len = min(self.sliding_window + self.num_memory_tokens, self.max_cache_len)
        
        if cache_position + seq_len > effective_cache_len:
            # Implement sliding window: keep memory tokens + recent window
            self._apply_sliding_window(cache_key, cache_position, seq_len, effective_cache_len)
            
        # Update cache
        end_pos = min(cache_position + seq_len, effective_cache_len)
        actual_seq_len = end_pos - cache_position
        
        if actual_seq_len > 0:
            self.key_cache[cache_key][:, :, cache_position:end_pos] = key_states[:, :, :actual_seq_len]
            self.value_cache[cache_key][:, :, cache_position:end_pos] = value_states[:, :, :actual_seq_len]
        
        # Return full cached sequence
        cached_key = self.key_cache[cache_key][:, :, :end_pos]
        cached_value = self.value_cache[cache_key][:, :, :end_pos]
        
        return cached_key, cached_value
        
    def _initialize_cache(self, cache_key: int, batch_size: int, num_heads: int, head_dim: int, device: torch.device, dtype: torch.dtype):
        """Initialize cache tensors for a cache key"""
        cache_shape = (batch_size, num_heads, self.max_cache_len, head_dim)
        self.key_cache[cache_key] = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.value_cache[cache_key] = torch.zeros(cache_shape, device=device, dtype=dtype)
        
    def _apply_sliding_window(self, cache_key: int, cache_position: int, seq_len: int, effective_cache_len: int):
        """Apply sliding window to cache when it exceeds capacity"""
        if cache_key not in self.key_cache:
            return
            
        # Keep memory tokens (first num_memory_tokens positions)
        # Slide the rest of the cache
        memory_end = self.num_memory_tokens
        window_size = effective_cache_len - memory_end
        
        if cache_position >= effective_cache_len:
            # Shift cache content
            shift_amount = cache_position + seq_len - effective_cache_len + 1
            
            # Move recent content to the beginning of the sliding window
            self.key_cache[cache_key][:, :, memory_end:memory_end + window_size - shift_amount] = \
                self.key_cache[cache_key][:, :, memory_end + shift_amount:effective_cache_len]
            self.value_cache[cache_key][:, :, memory_end:memory_end + window_size - shift_amount] = \
                self.value_cache[cache_key][:, :, memory_end + shift_amount:effective_cache_len]
                
            # Clear the end of cache
            self.key_cache[cache_key][:, :, memory_end + window_size - shift_amount:] = 0
            self.value_cache[cache_key][:, :, memory_end + window_size - shift_amount:] = 0
            
    def get_cache(self, layer_idx: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Get cached key and value states for a layer"""
        if not self.use_cache:
            return None, None
            
        cache_key = self.get_cache_key(layer_idx)
        
        if cache_key in self.key_cache:
            return self.key_cache[cache_key], self.value_cache[cache_key]
        return None, None
        
    def clear_cache(self):
        """Clear all cached states"""
        self.key_cache.clear()
        self.value_cache.clear()
        self.cache_position = 0
        
    def advance_cache_position(self, seq_len: int):
        """Advance the cache position by seq_len"""
        self.cache_position += seq_len
        
    def get_cache_size(self) -> Dict[str, int]:
        """Get cache size statistics"""
        total_elements = 0
        total_memory_mb = 0
        
        for cache_key in self.key_cache:
            key_elements = self.key_cache[cache_key].numel()
            value_elements = self.value_cache[cache_key].numel()
            total_elements += key_elements + value_elements
            
            # Estimate memory usage (assuming float16 = 2 bytes)
            total_memory_mb += (key_elements + value_elements) * 2 / (1024 * 1024)
            
        return {
            'total_elements': total_elements,
            'total_memory_mb': total_memory_mb,
            'num_cache_keys': len(self.key_cache),
            'cache_position': self.cache_position
        }


class CacheAwareAttention(nn.Module):
    """
    Attention module that integrates with HymbaCache
    """
    
    def __init__(self, attention_module, cache: HymbaCache, layer_idx: int):
        super().__init__()
        self.attention = attention_module
        self.cache = cache
        self.layer_idx = layer_idx
        
    def forward(self, query_states, key_states, value_states, cache_position=None, use_cache=True):
        """Forward pass with cache integration"""
        if use_cache and self.cache.use_cache:
            # Update cache and get full cached sequence
            key_states, value_states = self.cache.update_cache(
                self.layer_idx, key_states, value_states, cache_position
            )
            
        # Run attention with (potentially cached) key/value states
        return self.attention(query_states, key_states, value_states)


def create_cache_from_config(config) -> HymbaCache:
    """Factory function to create cache from configuration"""
    return HymbaCache(config)
