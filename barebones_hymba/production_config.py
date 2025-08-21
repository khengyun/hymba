"""
Production-ready Hymba Configuration
Matches HuggingFace model specifications with all advanced features
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Union
import json


@dataclass
class ProductionHymbaConfig:
    """
    Production configuration for Hymba model matching HuggingFace specs
    """
    
    # Model Architecture
    model_type: str = "hymba"
    architectures: List[str] = field(default_factory=lambda: ["HymbaForCausalLM"])
    
    # Core Parameters (scaled up from barebones)
    hidden_size: int = 1600  # vs 1536 in barebones
    num_hidden_layers: int = 32
    num_attention_heads: int = 25  # vs 12 in barebones
    num_key_value_heads: int = 5   # vs 2 in barebones
    vocab_size: int = 32001        # vs 151936 in barebones
    
    # Intermediate sizes
    intermediate_size: int = 5504  # MLP hidden size
    attn_hidden_size: int = -1     # Auto-calculate if -1
    
    # Memory Tokens (optimized)
    num_memory_tokens: int = 128   # vs 256 in barebones
    memory_tokens_interspersed_every: int = 0
    
    # Attention Configuration
    sliding_window: int = 1024     # vs 2048 in barebones
    global_attn_idx: List[int] = field(default_factory=lambda: [0, 15, 31])  # vs [5,11,18,25,31]
    attention_dropout: float = 0.0
    
    # Attention Implementation
    attn_implementation: str = "flex"
    attn_implementation_new: str = "flex"
    
    # SSM Parameters (scaled up)
    mamba_expand: int = 2
    mamba_d_conv: int = 4          # vs 3 in barebones
    mamba_d_state: int = 16
    mamba_dt_rank: int = 100       # vs 8 in barebones
    mamba_conv_bias: bool = True
    mamba_proj_bias: bool = False
    mamba_inner_layernorms: bool = True
    use_mamba_kernels: bool = True
    
    # Cross-layer KV Sharing (critical feature)
    kv_reuse_group: List[List[int]] = field(default_factory=lambda: [
        [1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14],
        [16, 17, 18], [19, 20], [21, 22], [23, 24], [25, 26], [27, 28], [29, 30]
    ])
    kv_weight_reuse: bool = False
    kv_reuse_every_i_layer: int = -1
    
    # Cache Management
    use_cache: bool = True
    calc_logits_for_entire_prompt: bool = False
    
    # Position Embeddings
    max_position_embeddings: int = 8192
    orig_max_position_embeddings: int = 2048
    rope: bool = True
    rope_theta: float = 10000.0
    rope_type: str = "ntk"
    
    # Layer Configuration
    layer_type: List[str] = field(default_factory=lambda: ["h"] * 32)
    
    # Per-layer convolution dimensions
    conv_dim: Dict[str, int] = field(default_factory=lambda: {str(i): 3200 for i in range(32)})
    
    # Normalization
    rms_norm_eps: float = 1e-06
    
    # Head dimensions
    kq_head_dim: int = -1  # Auto-calculate if -1
    v_head_dim: int = 128
    kq_norm: str = "none"
    
    # Activation functions
    hidden_act: str = "silu"
    mlp_hidden_act: str = "silu"
    
    # Token IDs (critical for generation)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    
    # Training parameters
    initializer_range: float = 0.02
    tie_word_embeddings: bool = True
    
    # Sequence length
    seq_length: int = 8192
    
    # MoE parameters (for future extension)
    num_experts: int = 1
    num_experts_per_tok: int = 1
    num_mamba: int = 1
    output_router_logits: bool = False
    router_aux_loss_coef: float = 0.001
    
    # Precision
    torch_dtype: str = "bfloat16"
    
    # Transformers version compatibility
    transformers_version: str = "4.44.0"
    
    def __post_init__(self):
        """Post-initialization validation and auto-calculation"""
        # Auto-calculate head dimensions if not specified
        if self.attn_hidden_size == -1:
            self.attn_hidden_size = self.hidden_size
            
        if self.kq_head_dim == -1:
            self.kq_head_dim = self.hidden_size // self.num_attention_heads
            
        # Validate configuration
        self._validate_config()
        
    def _validate_config(self):
        """Validate configuration parameters"""
        assert self.hidden_size % self.num_attention_heads == 0, \
            f"hidden_size ({self.hidden_size}) must be divisible by num_attention_heads ({self.num_attention_heads})"
            
        assert self.num_attention_heads % self.num_key_value_heads == 0, \
            f"num_attention_heads ({self.num_attention_heads}) must be divisible by num_key_value_heads ({self.num_key_value_heads})"
            
        assert self.sliding_window > 0, "sliding_window must be positive"
        assert self.num_memory_tokens >= 0, "num_memory_tokens must be non-negative"
        
        # Validate global attention indices
        for idx in self.global_attn_idx:
            assert 0 <= idx < self.num_hidden_layers, \
                f"global_attn_idx contains invalid layer index: {idx}"
                
        # Validate KV reuse groups
        all_layers_in_groups = set()
        for group in self.kv_reuse_group:
            for layer_idx in group:
                assert 0 <= layer_idx < self.num_hidden_layers, \
                    f"kv_reuse_group contains invalid layer index: {layer_idx}"
                assert layer_idx not in all_layers_in_groups, \
                    f"Layer {layer_idx} appears in multiple KV reuse groups"
                all_layers_in_groups.add(layer_idx)
                
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items() 
            if not k.startswith('_')
        }
        
    def save_pretrained(self, save_directory: str):
        """Save configuration to directory"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        config_file = os.path.join(save_directory, "config.json")
        with open(config_file, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
            
    @classmethod
    def from_dict(cls, config_dict: Dict):
        """Create config from dictionary"""
        return cls(**config_dict)
        
    @classmethod
    def from_pretrained(cls, model_name_or_path: str):
        """Load configuration from pretrained model"""
        import os
        
        if os.path.isdir(model_name_or_path):
            config_file = os.path.join(model_name_or_path, "config.json")
        else:
            # Handle HuggingFace model names
            from huggingface_hub import hf_hub_download
            config_file = hf_hub_download(model_name_or_path, "config.json")
            
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
            
        return cls.from_dict(config_dict)
        
    def get_cache_config(self) -> Dict:
        """Get cache-specific configuration"""
        return {
            'use_cache': self.use_cache,
            'sliding_window': self.sliding_window,
            'num_memory_tokens': self.num_memory_tokens,
            'kv_reuse_group': self.kv_reuse_group,
            'max_position_embeddings': self.max_position_embeddings,
            'hidden_size': self.hidden_size,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads
        }


# Backward compatibility with barebones config
class BarebonesHymbaConfig:
    """Barebones configuration for comparison"""
    
    def __init__(self):
        self.hidden_size = 1536
        self.num_attention_heads = 12
        self.num_key_value_heads = 2
        self.vocab_size = 151936
        self.num_hidden_layers = 32
        self.num_meta_tokens = 256
        self.attention_window_size = 2048
        self.global_layer_list = [5, 11, 18, 25, 31]
        self.mamba_expand = 2
        self.time_step_rank = 8
        self.conv_kernel_size = 3
        self.ssm_state_size = 16
        self.modify_attention_mask = True
        self.seq_length = 4096


def create_production_config() -> ProductionHymbaConfig:
    """Factory function to create production configuration"""
    return ProductionHymbaConfig()


def create_barebones_config() -> BarebonesHymbaConfig:
    """Factory function to create barebones configuration"""
    return BarebonesHymbaConfig()


def compare_configs():
    """Compare barebones vs production configurations"""
    barebones = create_barebones_config()
    production = create_production_config()
    
    print("Configuration Comparison:")
    print("=" * 50)
    
    comparisons = [
        ("hidden_size", barebones.hidden_size, production.hidden_size),
        ("num_attention_heads", barebones.num_attention_heads, production.num_attention_heads),
        ("num_key_value_heads", barebones.num_key_value_heads, production.num_key_value_heads),
        ("vocab_size", barebones.vocab_size, production.vocab_size),
        ("memory_tokens", barebones.num_meta_tokens, production.num_memory_tokens),
        ("sliding_window", barebones.attention_window_size, production.sliding_window),
        ("mamba_dt_rank", barebones.time_step_rank, production.mamba_dt_rank),
    ]
    
    for name, bare_val, prod_val in comparisons:
        status = "✅ Match" if bare_val == prod_val else "⚠️ Different"
        print(f"{name:20} | {bare_val:>8} | {prod_val:>8} | {status}")


if __name__ == "__main__":
    # Test configuration
    config = create_production_config()
    print("Production config created successfully!")
    print(f"Model type: {config.model_type}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Cache enabled: {config.use_cache}")
    
    # Compare configurations
    compare_configs()
