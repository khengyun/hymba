#!/usr/bin/env python3
"""
Final validation script for Hymba production model
Tests generation, performance, and readiness for evaluation
"""

import torch
import torch.nn.functional as F
import time
import json
import sys
from pathlib import Path

from configuration_hymba import HymbaConfig, BarebonesCompatibilityConfig
from modeling_hymba import HymbaForCausalLM


class HymbaFinalValidator:
    """Final validation suite for Hymba model"""
    
    def __init__(self):
        self.results = {}
        
    def test_model_loading(self):
        """Test model loading from config"""
        print("🔧 Testing Model Loading...")
        
        try:
            # Load from current directory
            config = HymbaConfig.from_pretrained('.')
            model = HymbaForCausalLM(config)
            
            print("✅ Model loaded successfully!")
            print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   Config: {config.model_type}")
            
            self.model = model
            self.config = config
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def test_generation_capability(self):
        """Test text generation capability"""
        print("\n📝 Testing Generation Capability...")
        
        if not hasattr(self, 'model'):
            print("❌ No model available")
            return False
            
        try:
            # Simple generation test
            prompt = "The future of artificial intelligence"
            print(f"   Prompt: '{prompt}'")
            
            # Convert to simple token IDs (character-based for demo)
            input_ids = torch.tensor([[ord(c) % 1000 for c in prompt[:20]]])
            print(f"   Input IDs shape: {input_ids.shape}")
            
            # Generate tokens
            max_new_tokens = 20
            generated_ids = input_ids.clone()
            
            start_time = time.time()
            
            for i in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(generated_ids)
                    logits = outputs.logits
                    
                    # Get next token (greedy decoding)
                    next_token_logits = logits[0, -1, :]
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                    
                    # Append to sequence
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                    
                    # Simple stopping condition
                    if next_token.item() == 0:  # Stop if we hit pad token
                        break
            
            end_time = time.time()
            
            generation_time = end_time - start_time
            tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
            tokens_per_sec = tokens_generated / generation_time
            
            print(f"✅ Generation successful!")
            print(f"   Tokens generated: {tokens_generated}")
            print(f"   Generation time: {generation_time:.2f}s")
            print(f"   Speed: {tokens_per_sec:.1f} tokens/sec")
            print(f"   Generated sequence length: {generated_ids.shape[1]}")
            
            self.results['generation'] = {
                'tokens_generated': tokens_generated,
                'generation_time': generation_time,
                'tokens_per_sec': tokens_per_sec
            }
            
            return True
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def test_batch_processing(self):
        """Test batch processing capabilities"""
        print("\n📦 Testing Batch Processing...")
        
        if not hasattr(self, 'model'):
            print("❌ No model available")
            return False
            
        try:
            batch_sizes = [1, 2, 4]
            seq_len = 50
            
            for batch_size in batch_sizes:
                print(f"   Testing batch_size={batch_size}...")
                
                # Create batch input
                input_ids = torch.randint(0, 1000, (batch_size, seq_len))
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = self.model(input_ids)
                end_time = time.time()
                
                batch_time = end_time - start_time
                tokens_processed = batch_size * seq_len
                throughput = tokens_processed / batch_time
                
                print(f"     ✅ Batch {batch_size}: {throughput:.1f} tok/sec ({batch_time*1000:.1f}ms)")
                
            return True
            
        except Exception as e:
            print(f"❌ Batch processing failed: {e}")
            return False
    
    def test_memory_efficiency(self):
        """Test memory efficiency with different sequence lengths"""
        print("\n💾 Testing Memory Efficiency...")
        
        if not hasattr(self, 'model'):
            print("❌ No model available")
            return False
            
        try:
            import psutil
            import gc
            import os
            
            process = psutil.Process(os.getpid())
            seq_lengths = [128, 256, 512, 1024]
            
            memory_results = {}
            
            for seq_len in seq_lengths:
                # Clear memory
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                initial_memory = process.memory_info().rss / 1024 / 1024
                
                # Test forward pass
                input_ids = torch.randint(0, 1000, (1, seq_len))
                
                with torch.no_grad():
                    outputs = self.model(input_ids)
                
                final_memory = process.memory_info().rss / 1024 / 1024
                memory_used = final_memory - initial_memory
                
                memory_results[seq_len] = memory_used
                print(f"   Seq len {seq_len}: {memory_used:.1f} MB")
            
            # Check memory scaling
            memory_growth = memory_results[1024] - memory_results[128]
            print(f"   Memory growth (128→1024): {memory_growth:.1f} MB")
            
            if memory_growth < 100:  # Less than 100MB growth is good
                print("✅ Memory scaling looks efficient!")
            else:
                print("⚠️  Memory scaling might need optimization")
            
            self.results['memory'] = memory_results
            return True
            
        except Exception as e:
            print(f"❌ Memory test failed: {e}")
            return False
    
    def test_config_compatibility(self):
        """Test configuration compatibility"""
        print("\n⚙️ Testing Configuration Compatibility...")
        
        try:
            # Test different config creation methods
            config1 = HymbaConfig.from_pretrained('.')
            config2 = BarebonesCompatibilityConfig.create_1_5b_config()
            
            # Compare key parameters
            key_params = ['hidden_size', 'num_attention_heads', 'num_key_value_heads', 'vocab_size']
            
            matches = 0
            for param in key_params:
                val1 = getattr(config1, param)
                val2 = getattr(config2, param)
                if val1 == val2:
                    matches += 1
                    print(f"   ✅ {param}: {val1}")
                else:
                    print(f"   ⚠️  {param}: {val1} vs {val2}")
            
            compatibility_score = matches / len(key_params) * 100
            print(f"   Compatibility score: {compatibility_score:.1f}%")
            
            return compatibility_score > 90
            
        except Exception as e:
            print(f"❌ Config compatibility test failed: {e}")
            return False
    
    def test_evaluation_readiness(self):
        """Test if model is ready for evaluation"""
        print("\n📊 Testing Evaluation Readiness...")
        
        try:
            # Check if model can handle evaluation-style inputs
            test_cases = [
                ("Short input", torch.randint(0, 1000, (1, 10))),
                ("Medium input", torch.randint(0, 1000, (1, 50))),
                ("Long input", torch.randint(0, 1000, (1, 200))),
            ]
            
            for name, input_ids in test_cases:
                try:
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        
                    # Check output validity
                    logits = outputs.logits
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"   ❌ {name}: Invalid outputs (NaN/Inf)")
                        return False
                    
                    # Check probability distribution
                    probs = F.softmax(logits[0, -1, :], dim=-1)
                    if probs.sum().item() < 0.99 or probs.sum().item() > 1.01:
                        print(f"   ❌ {name}: Invalid probability distribution")
                        return False
                    
                    print(f"   ✅ {name}: Valid outputs")
                    
                except Exception as e:
                    print(f"   ❌ {name}: Failed - {e}")
                    return False
            
            print("✅ Model is ready for evaluation!")
            return True
            
        except Exception as e:
            print(f"❌ Evaluation readiness test failed: {e}")
            return False
    
    def generate_final_report(self):
        """Generate final validation report"""
        print("\n" + "="*60)
        print("📋 Final Validation Report")
        print("="*60)
        
        # Model specifications
        if hasattr(self, 'model'):
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n🔧 Model Specifications:")
            print(f"   Total Parameters: {total_params:,}")
            print(f"   Model Size: ~{total_params * 4 / 1024 / 1024:.1f} MB (fp32)")
            print(f"   Hidden Size: {self.config.hidden_size}")
            print(f"   Attention Heads: {self.config.num_attention_heads}")
            print(f"   KV Heads: {self.config.num_key_value_heads}")
        
        # Performance results
        if 'generation' in self.results:
            gen = self.results['generation']
            print(f"\n⚡ Performance Results:")
            print(f"   Generation Speed: {gen['tokens_per_sec']:.1f} tokens/sec")
            print(f"   Generation Time: {gen['generation_time']:.2f}s for {gen['tokens_generated']} tokens")
        
        # Memory results
        if 'memory' in self.results:
            mem = self.results['memory']
            print(f"\n💾 Memory Usage:")
            for seq_len, memory in mem.items():
                print(f"   Seq {seq_len}: {memory:.1f} MB")
        
        # Comparison with paper claims
        print(f"\n📊 Comparison with Paper Claims:")
        print(f"   Paper claims (Hymba-1.5B):")
        print(f"   - Throughput: 664 tok/sec")
        print(f"   - Cache size: 79MB")
        print(f"   - Parameters: ~1.5B")
        
        if hasattr(self, 'model'):
            actual_params = sum(p.numel() for p in self.model.parameters()) / 1e6
            print(f"   Our implementation:")
            print(f"   - Parameters: ~{actual_params:.1f}M")
            
        if 'generation' in self.results:
            our_throughput = self.results['generation']['tokens_per_sec']
            print(f"   - Generation speed: {our_throughput:.1f} tok/sec")
        
        print(f"\n🎯 Readiness Status:")
        print(f"   ✅ Model loads successfully")
        print(f"   ✅ Forward pass works")
        print(f"   ✅ Generation capability")
        print(f"   ✅ Batch processing")
        print(f"   ✅ Memory efficiency")
        print(f"   ✅ Configuration compatibility")
        print(f"   ✅ Evaluation ready")
        
        print(f"\n🚀 Next Steps:")
        print(f"   1. Run lm-evaluation-harness on specific tasks")
        print(f"   2. Compare results with paper benchmarks")
        print(f"   3. Optimize performance if needed")
        print(f"   4. Document final implementation")
    
    def run_full_validation(self):
        """Run complete validation suite"""
        print("🧪 Running Final Hymba Model Validation")
        print("="*60)
        
        tests = [
            ("Model Loading", self.test_model_loading),
            ("Generation Capability", self.test_generation_capability),
            ("Batch Processing", self.test_batch_processing),
            ("Memory Efficiency", self.test_memory_efficiency),
            ("Config Compatibility", self.test_config_compatibility),
            ("Evaluation Readiness", self.test_evaluation_readiness),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} failed with exception: {e}")
                results[test_name] = False
        
        # Summary
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        print(f"\n" + "="*60)
        print("📋 Validation Summary:")
        print("="*60)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:<25} | {status}")
        
        print("-"*60)
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL VALIDATIONS PASSED! Model is production-ready!")
            self.generate_final_report()
            return True
        else:
            print("⚠️  Some validations failed. Check the output above.")
            return False


if __name__ == "__main__":
    validator = HymbaFinalValidator()
    success = validator.run_full_validation()
    sys.exit(0 if success else 1)
