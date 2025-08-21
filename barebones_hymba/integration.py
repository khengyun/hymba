"""
Integration script to connect production Hymba model with barebones implementation
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

# Import production components
from configuration_hymba import HymbaConfig, BarebonesCompatibilityConfig
from modeling_hymba import HymbaForCausalLM, HybridMambaAttentionDynamicCache

# Import barebones components
from test_barebones_hymba import SimplifiedHymba, HymbaConfig as BarebonesConfig
from barebones_hymba_block import HymbaBlock


class IntegratedHymbaForCausalLM(HymbaForCausalLM):
    """
    Enhanced HymbaForCausalLM that integrates barebones implementation
    with production features
    """
    
    def __init__(self, config: HymbaConfig):
        super().__init__(config)
        
        # Replace simplified model with barebones implementation
        self.model = self._create_barebones_model(config)
        
    def _create_barebones_model(self, config: HymbaConfig):
        """Create barebones model with production config"""
        
        # Convert production config to barebones format
        barebones_config = self._convert_to_barebones_config(config)
        
        # Create barebones model
        model = SimplifiedHymba(barebones_config)
        
        return model
        
    def _convert_to_barebones_config(self, prod_config: HymbaConfig):
        """Convert production config to barebones config format"""
        
        class BarebonesConfigAdapter:
            def __init__(self, prod_config):
                # Map production parameters to barebones
                self.hidden_size = prod_config.hidden_size
                self.num_attention_heads = prod_config.num_attention_heads
                self.num_key_value_heads = prod_config.num_key_value_heads
                self.vocab_size = prod_config.vocab_size
                self.num_hidden_layers = prod_config.num_hidden_layers
                self.num_meta_tokens = prod_config.num_memory_tokens
                self.attention_window_size = prod_config.sliding_window
                self.global_layer_list = prod_config.global_attn_idx or [5, 11, 18, 25, 31]
                self.mamba_expand = prod_config.mamba_expand
                self.time_step_rank = prod_config.mamba_dt_rank
                self.conv_kernel_size = prod_config.mamba_d_conv
                self.ssm_state_size = prod_config.mamba_d_state
                self.intermediate_size = prod_config.intermediate_size
                self.mlp_hidden_act = prod_config.hidden_act
                self.modify_attention_mask = True
                self.seq_length = prod_config.max_position_embeddings
                self.use_positional_embedding = prod_config.rope
                self.rope_base = getattr(prod_config, 'rope_theta', 10000)
                
        return BarebonesConfigAdapter(prod_config)
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Any] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        calc_logits_for_entire_prompt: Optional[bool] = True,
    ):
        """Enhanced forward pass with barebones model"""
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
        # Run through barebones model
        hidden_states = self.model(inputs_embeds)
        
        # Apply final layernorm
        hidden_states = self.final_layernorm(hidden_states)
        
        # Calculate logits
        if calc_logits_for_entire_prompt:
            logits = self.lm_head(hidden_states)
        else:
            logits = self.lm_head(hidden_states[..., -1:, :])
        logits = logits.float()

        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
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


class ProductionModelFactory:
    """Factory class to create different model configurations"""
    
    @staticmethod
    def create_1_5b_model():
        """Create Hymba-1.5B model"""
        config = BarebonesCompatibilityConfig.create_1_5b_config()
        return IntegratedHymbaForCausalLM(config)
        
    @staticmethod
    def create_from_barebones_config(barebones_config):
        """Create production model from barebones config"""
        prod_config = BarebonesCompatibilityConfig.from_barebones_to_production(barebones_config)
        return IntegratedHymbaForCausalLM(prod_config)
        
    @staticmethod
    def create_from_pretrained(model_name_or_path="nvidia/Hymba-1.5B-Instruct"):
        """Create model from pretrained checkpoint"""
        try:
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
            return IntegratedHymbaForCausalLM(config)
        except Exception as e:
            print(f"Failed to load from {model_name_or_path}: {e}")
            print("Falling back to 1.5B config")
            return ProductionModelFactory.create_1_5b_model()


def test_integration():
    """Test the integration between barebones and production"""
    
    print("Testing Hymba Integration...")
    
    # Test 1: Create model from barebones config
    print("\n1. Testing barebones config conversion...")
    barebones_config = BarebonesConfig()
    model = ProductionModelFactory.create_from_barebones_config(barebones_config)
    print(f"✅ Model created with hidden_size: {model.config.hidden_size}")
    
    # Test 2: Create 1.5B model
    print("\n2. Testing 1.5B model creation...")
    model_1_5b = ProductionModelFactory.create_1_5b_model()
    print(f"✅ 1.5B model created with hidden_size: {model_1_5b.config.hidden_size}")
    
    # Test 3: Forward pass
    print("\n3. Testing forward pass...")
    batch_size = 1
    seq_len = 10
    vocab_size = model_1_5b.config.vocab_size
    
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model_1_5b(input_ids)
        
    print(f"✅ Forward pass successful!")
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Output logits shape: {outputs.logits.shape}")
    print(f"   Expected shape: {(batch_size, seq_len, vocab_size)}")
    
    # Test 4: Generation preparation
    print("\n4. Testing generation preparation...")
    gen_inputs = model_1_5b.prepare_inputs_for_generation(input_ids)
    print(f"✅ Generation inputs prepared: {list(gen_inputs.keys())}")
    
    print("\n🎉 All integration tests passed!")
    
    return model_1_5b


def compare_configs():
    """Compare barebones vs production configurations"""
    
    print("Configuration Comparison:")
    print("=" * 60)
    
    # Create configs
    barebones = BarebonesConfig()
    production = BarebonesCompatibilityConfig.create_1_5b_config()
    
    # Compare key parameters
    comparisons = [
        ("hidden_size", getattr(barebones, 'hidden_size', 'N/A'), production.hidden_size),
        ("num_attention_heads", getattr(barebones, 'num_attention_heads', 'N/A'), production.num_attention_heads),
        ("num_key_value_heads", getattr(barebones, 'num_key_value_heads', 'N/A'), production.num_key_value_heads),
        ("vocab_size", getattr(barebones, 'vocab_size', 'N/A'), production.vocab_size),
        ("memory_tokens", getattr(barebones, 'num_meta_tokens', 'N/A'), production.num_memory_tokens),
        ("sliding_window", getattr(barebones, 'attention_window_size', 'N/A'), production.sliding_window),
        ("mamba_dt_rank", getattr(barebones, 'time_step_rank', 'N/A'), production.mamba_dt_rank),
        ("use_cache", "False", production.use_cache),
        ("rope", getattr(barebones, 'use_positional_embedding', 'N/A'), production.rope),
    ]
    
    print(f"{'Parameter':<20} | {'Barebones':<12} | {'Production':<12} | {'Status'}")
    print("-" * 60)
    
    for name, bare_val, prod_val in comparisons:
        if bare_val == prod_val:
            status = "✅ Match"
        elif bare_val == 'N/A':
            status = "➕ New"
        else:
            status = "⚠️ Different"
        print(f"{name:<20} | {str(bare_val):<12} | {str(prod_val):<12} | {status}")


if __name__ == "__main__":
    # Run tests
    compare_configs()
    print("\n" + "="*60 + "\n")
    test_integration()
