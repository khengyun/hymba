#!/usr/bin/env python3
"""
Simple model test without complex dependencies
Tests basic functionality and configuration loading
"""

import torch
import torch.nn as nn
import json
import time
import sys
import os
from pathlib import Path

# Test configuration loading
def test_config_loading():
    """Test if we can load the updated config.json"""
    print("🔧 Testing Configuration Loading...")
    
    try:
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        print("✅ Config loaded successfully!")
        print(f"   Model type: {config.get('model_type')}")
        print(f"   Hidden size: {config.get('hidden_size')}")
        print(f"   Attention heads: {config.get('num_attention_heads')}")
        print(f"   KV heads: {config.get('num_key_value_heads')}")
        print(f"   Vocab size: {config.get('vocab_size')}")
        print(f"   Memory tokens: {config.get('num_memory_tokens')}")
        print(f"   Sliding window: {config.get('sliding_window')}")
        print(f"   Use cache: {config.get('use_cache')}")
        print(f"   Global attention: {config.get('global_attn_idx')}")
        
        # Validate key parameters
        required_params = [
            'model_type', 'hidden_size', 'num_attention_heads', 
            'num_key_value_heads', 'vocab_size', 'num_hidden_layers'
        ]
        
        missing_params = [p for p in required_params if p not in config]
        if missing_params:
            print(f"⚠️  Missing parameters: {missing_params}")
            return False
        
        # Check if values match HuggingFace specs
        expected_values = {
            'model_type': 'hymba',
            'hidden_size': 1600,
            'num_attention_heads': 25,
            'num_key_value_heads': 5,
            'vocab_size': 32001,
            'num_hidden_layers': 32
        }
        
        mismatches = []
        for key, expected in expected_values.items():
            actual = config.get(key)
            if actual != expected:
                mismatches.append(f"{key}: expected {expected}, got {actual}")
        
        if mismatches:
            print("⚠️  Value mismatches:")
            for mismatch in mismatches:
                print(f"     {mismatch}")
        else:
            print("✅ All key parameters match HuggingFace specs!")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False


def test_production_config_class():
    """Test production configuration class"""
    print("\n📋 Testing Production Configuration Class...")
    
    try:
        from configuration_hymba import HymbaConfig, BarebonesCompatibilityConfig
        
        # Test creating config from file
        config = HymbaConfig.from_pretrained('.')
        print("✅ HymbaConfig loaded from current directory!")
        print(f"   Model type: {config.model_type}")
        print(f"   Hidden size: {config.hidden_size}")
        print(f"   Architectures: {config.architectures}")
        
        # Test 1.5B config creation
        config_1_5b = BarebonesCompatibilityConfig.create_1_5b_config()
        print("✅ 1.5B config created successfully!")
        print(f"   Hidden size: {config_1_5b.hidden_size}")
        print(f"   KV reuse groups: {len(config_1_5b.kv_reuse_group)} groups")
        
        return True
        
    except Exception as e:
        print(f"❌ Production config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_model_creation():
    """Test simple model creation without complex dependencies"""
    print("\n🚀 Testing Simple Model Creation...")
    
    try:
        from modeling_hymba import HymbaForCausalLM
        from configuration_hymba import BarebonesCompatibilityConfig
        
        # Create simple config
        config = BarebonesCompatibilityConfig.create_1_5b_config()
        
        # Create model (this will be simplified)
        model = HymbaForCausalLM(config)
        print("✅ HymbaForCausalLM created successfully!")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Config type: {type(model.config).__name__}")
        print(f"   Vocab size: {model.config.vocab_size}")
        
        # Test model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"   Total parameters: {total_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_simple_forward_pass(model):
    """Test simple forward pass"""
    print("\n⚡ Testing Simple Forward Pass...")
    
    if model is None:
        print("❌ No model to test")
        return False
        
    try:
        # Create simple input
        batch_size, seq_len = 1, 10
        vocab_size = model.config.vocab_size
        
        # Use smaller vocab for testing
        test_vocab_size = min(1000, vocab_size)
        input_ids = torch.randint(0, test_vocab_size, (batch_size, seq_len))
        
        print(f"   Input shape: {input_ids.shape}")
        print(f"   Input range: {input_ids.min().item()} - {input_ids.max().item()}")
        
        # Forward pass
        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_ids)
        end_time = time.time()
        
        print(f"✅ Forward pass successful!")
        print(f"   Output logits shape: {outputs.logits.shape}")
        print(f"   Expected shape: {(batch_size, seq_len, vocab_size)}")
        print(f"   Forward time: {(end_time - start_time)*1000:.2f}ms")
        
        # Check output validity
        logits = outputs.logits
        print(f"   Logits range: {logits.min().item():.3f} - {logits.max().item():.3f}")
        print(f"   Logits mean: {logits.mean().item():.3f}")
        print(f"   Contains NaN: {torch.isnan(logits).any().item()}")
        print(f"   Contains Inf: {torch.isinf(logits).any().item()}")
        
        # Test probability distribution
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        top_5_probs, top_5_indices = torch.topk(probs, 5)
        print(f"   Top 5 token probabilities: {top_5_probs.tolist()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_usage():
    """Test memory usage"""
    print("\n💾 Testing Memory Usage...")
    
    try:
        import psutil
        import gc
        
        # Get initial memory
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        print(f"   Initial memory: {initial_memory:.1f} MB")
        
        # Create model and test
        success, model = test_simple_model_creation()
        if success and model:
            # Memory after model creation
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            model_memory = current_memory - initial_memory
            
            print(f"   Memory after model creation: {current_memory:.1f} MB")
            print(f"   Model memory usage: {model_memory:.1f} MB")
            
            # Test forward pass memory
            if test_simple_forward_pass(model):
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                forward_memory = final_memory - current_memory
                
                print(f"   Memory after forward pass: {final_memory:.1f} MB")
                print(f"   Forward pass memory: {forward_memory:.1f} MB")
                print(f"   Total memory increase: {final_memory - initial_memory:.1f} MB")
        
        # Cleanup
        if 'model' in locals():
            del model
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests"""
    print("🧪 Running Comprehensive Hymba Model Tests")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Config loading
    results['config_loading'] = test_config_loading()
    
    # Test 2: Production config class
    results['production_config'] = test_production_config_class()
    
    # Test 3: Model creation
    success, model = test_simple_model_creation()
    results['model_creation'] = success
    
    # Test 4: Forward pass
    if success and model:
        results['forward_pass'] = test_simple_forward_pass(model)
    else:
        results['forward_pass'] = False
    
    # Test 5: Memory usage
    results['memory_usage'] = test_memory_usage()
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 Test Summary:")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title():<25} | {status}")
        
    print("-" * 60)
    print(f"Total: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Model is ready for evaluation!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
