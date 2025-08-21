#!/usr/bin/env python3
"""
Test script for production Hymba integration
"""

import torch
import torch.nn.functional as F
import time
import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Change to barebones_hymba directory for imports
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from integration import ProductionModelFactory, compare_configs
from configuration_hymba import BarebonesCompatibilityConfig


def test_model_creation():
    """Test different ways of creating the model"""
    print("🔧 Testing Model Creation...")
    
    # Test 1: Create 1.5B model
    print("\n1. Creating Hymba-1.5B model...")
    try:
        model = ProductionModelFactory.create_1_5b_model()
        print(f"✅ Success! Model config:")
        print(f"   - Hidden size: {model.config.hidden_size}")
        print(f"   - Attention heads: {model.config.num_attention_heads}")
        print(f"   - KV heads: {model.config.num_key_value_heads}")
        print(f"   - Vocab size: {model.config.vocab_size}")
        print(f"   - Memory tokens: {model.config.num_memory_tokens}")
        print(f"   - Cache enabled: {model.config.use_cache}")
        return model
    except Exception as e:
        print(f"❌ Failed: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass with different input sizes"""
    print("\n🚀 Testing Forward Pass...")
    
    if model is None:
        print("❌ No model to test")
        return False
        
    try:
        # Test different sequence lengths
        test_cases = [
            (1, 10),   # Small sequence
            (2, 50),   # Medium sequence  
            (1, 100),  # Longer sequence
        ]
        
        for batch_size, seq_len in test_cases:
            print(f"\n   Testing batch_size={batch_size}, seq_len={seq_len}...")
            
            # Create random input
            input_ids = torch.randint(0, min(1000, model.config.vocab_size), (batch_size, seq_len))
            
            # Forward pass
            start_time = time.time()
            with torch.no_grad():
                outputs = model(input_ids)
            end_time = time.time()
            
            # Verify output shape
            expected_shape = (batch_size, seq_len, model.config.vocab_size)
            actual_shape = outputs.logits.shape
            
            if actual_shape == expected_shape:
                print(f"   ✅ Shape correct: {actual_shape}")
                print(f"   ⏱️  Time: {(end_time - start_time)*1000:.2f}ms")
            else:
                print(f"   ❌ Shape mismatch: expected {expected_shape}, got {actual_shape}")
                return False
                
        return True
        
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation_preparation(model):
    """Test generation input preparation"""
    print("\n📝 Testing Generation Preparation...")
    
    if model is None:
        print("❌ No model to test")
        return False
        
    try:
        # Create test input
        input_ids = torch.randint(0, min(1000, model.config.vocab_size), (1, 20))
        attention_mask = torch.ones_like(input_ids)
        
        # Prepare generation inputs
        gen_inputs = model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True
        )
        
        # Verify required keys
        required_keys = ['input_ids', 'past_key_values', 'use_cache', 'attention_mask']
        missing_keys = [key for key in required_keys if key not in gen_inputs]
        
        if not missing_keys:
            print(f"✅ All required keys present: {list(gen_inputs.keys())}")
            return True
        else:
            print(f"❌ Missing keys: {missing_keys}")
            return False
            
    except Exception as e:
        print(f"❌ Generation preparation failed: {e}")
        return False


def test_loss_calculation(model):
    """Test loss calculation with labels"""
    print("\n📊 Testing Loss Calculation...")
    
    if model is None:
        print("❌ No model to test")
        return False
        
    try:
        batch_size, seq_len = 2, 20
        vocab_size = model.config.vocab_size
        
        # Create input and labels
        input_ids = torch.randint(0, min(1000, vocab_size), (batch_size, seq_len))
        labels = torch.randint(0, min(1000, vocab_size), (batch_size, seq_len))
        
        # Forward pass with labels
        with torch.no_grad():
            outputs = model(input_ids=input_ids, labels=labels)
            
        # Check if loss is calculated
        if outputs.loss is not None:
            print(f"✅ Loss calculated: {outputs.loss.item():.4f}")
            print(f"   Logits shape: {outputs.logits.shape}")
            return True
        else:
            print("❌ Loss not calculated")
            return False
            
    except Exception as e:
        print(f"❌ Loss calculation failed: {e}")
        return False


def test_memory_usage(model):
    """Test memory usage and efficiency"""
    print("\n💾 Testing Memory Usage...")
    
    if model is None:
        print("❌ No model to test")
        return False
        
    try:
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        else:
            initial_memory = 0
            
        # Test with different sequence lengths
        seq_lengths = [50, 100, 200]
        memory_usage = []
        
        for seq_len in seq_lengths:
            input_ids = torch.randint(0, min(1000, model.config.vocab_size), (1, seq_len))
            
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                model = model.cuda()
                
            with torch.no_grad():
                outputs = model(input_ids)
                
            if torch.cuda.is_available():
                current_memory = torch.cuda.memory_allocated()
                memory_used = (current_memory - initial_memory) / 1024 / 1024  # MB
                memory_usage.append((seq_len, memory_used))
                print(f"   Seq len {seq_len}: {memory_used:.2f} MB")
            else:
                print(f"   Seq len {seq_len}: CPU mode (memory not tracked)")
                
        if torch.cuda.is_available() and memory_usage:
            # Check if memory usage is reasonable
            max_memory = max(usage[1] for usage in memory_usage)
            if max_memory < 1000:  # Less than 1GB
                print(f"✅ Memory usage reasonable (max: {max_memory:.2f} MB)")
                return True
            else:
                print(f"⚠️  High memory usage (max: {max_memory:.2f} MB)")
                return True
        else:
            print("✅ Memory test completed (CPU mode)")
            return True
            
    except Exception as e:
        print(f"❌ Memory test failed: {e}")
        return False


def run_comprehensive_test():
    """Run all tests"""
    print("🧪 Running Comprehensive Hymba Integration Tests")
    print("=" * 60)
    
    # Show configuration comparison
    compare_configs()
    print("\n" + "=" * 60)
    
    # Test results
    results = {}
    
    # Test 1: Model creation
    model = test_model_creation()
    results['model_creation'] = model is not None
    
    # Test 2: Forward pass
    results['forward_pass'] = test_forward_pass(model)
    
    # Test 3: Generation preparation
    results['generation_prep'] = test_generation_preparation(model)
    
    # Test 4: Loss calculation
    results['loss_calculation'] = test_loss_calculation(model)
    
    # Test 5: Memory usage
    results['memory_usage'] = test_memory_usage(model)
    
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
        print("🎉 All tests passed! Integration successful!")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above.")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    sys.exit(0 if success else 1)
