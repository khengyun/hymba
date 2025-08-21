#!/usr/bin/env python3
"""
Performance benchmarking for Hymba model
Tests throughput, memory usage, and basic functionality
"""

import torch
import torch.nn.functional as F
import time
import psutil
import os
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from integration import ProductionModelFactory
from configuration_hymba import BarebonesCompatibilityConfig


class HymbaPerformanceBenchmark:
    """Performance benchmark suite for Hymba model"""
    
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.results = {}
        
    def create_model(self):
        """Create model for benchmarking"""
        print("🔧 Creating Hymba model...")
        
        try:
            # Create 1.5B model
            model = ProductionModelFactory.create_1_5b_model()
            print(f"✅ Model created successfully!")
            print(f"   - Hidden size: {model.config.hidden_size}")
            print(f"   - Attention heads: {model.config.num_attention_heads}")
            print(f"   - Vocab size: {model.config.vocab_size}")
            return model
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            return None
    
    def benchmark_throughput(self, model, seq_lengths=[128, 512, 1024], batch_sizes=[1, 4]):
        """Benchmark throughput at different sequence lengths and batch sizes"""
        print("\n⚡ Benchmarking Throughput...")
        
        if model is None:
            print("❌ No model to benchmark")
            return {}
            
        throughput_results = {}
        
        for batch_size in batch_sizes:
            for seq_len in seq_lengths:
                print(f"\n   Testing batch_size={batch_size}, seq_len={seq_len}...")
                
                try:
                    # Create input
                    vocab_size = min(1000, model.config.vocab_size)  # Use smaller vocab for testing
                    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
                    
                    # Move to GPU if available
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                    if device == "cuda":
                        model = model.cuda()
                        input_ids = input_ids.cuda()
                    
                    # Warmup
                    print("     Warming up...")
                    for _ in range(3):
                        with torch.no_grad():
                            _ = model(input_ids)
                    
                    # Benchmark
                    print("     Benchmarking...")
                    num_runs = 10
                    
                    if device == "cuda":
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    
                    for _ in range(num_runs):
                        with torch.no_grad():
                            outputs = model(input_ids)
                    
                    if device == "cuda":
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    
                    # Calculate metrics
                    total_time = end_time - start_time
                    avg_time_per_run = total_time / num_runs
                    total_tokens = batch_size * seq_len * num_runs
                    throughput = total_tokens / total_time  # tokens/second
                    
                    result = {
                        'batch_size': batch_size,
                        'seq_len': seq_len,
                        'avg_time_ms': avg_time_per_run * 1000,
                        'throughput_tok_per_sec': throughput,
                        'device': device
                    }
                    
                    key = f"bs{batch_size}_seq{seq_len}"
                    throughput_results[key] = result
                    
                    print(f"     ✅ {throughput:.1f} tok/sec ({avg_time_per_run*1000:.1f}ms per run)")
                    
                except Exception as e:
                    print(f"     ❌ Failed: {e}")
                    continue
        
        self.results['throughput'] = throughput_results
        return throughput_results
    
    def benchmark_memory_usage(self, model, seq_lengths=[128, 512, 1024]):
        """Benchmark memory usage at different sequence lengths"""
        print("\n💾 Benchmarking Memory Usage...")
        
        if model is None:
            print("❌ No model to benchmark")
            return {}
            
        memory_results = {}
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda":
            model = model.cuda()
        
        for seq_len in seq_lengths:
            print(f"\n   Testing seq_len={seq_len}...")
            
            try:
                # Clear cache
                if device == "cuda":
                    torch.cuda.empty_cache()
                    initial_memory = torch.cuda.memory_allocated()
                else:
                    initial_memory = 0
                
                # Create input
                vocab_size = min(1000, model.config.vocab_size)
                input_ids = torch.randint(0, vocab_size, (1, seq_len))
                
                if device == "cuda":
                    input_ids = input_ids.cuda()
                
                # Forward pass
                with torch.no_grad():
                    outputs = model(input_ids)
                
                # Measure memory
                if device == "cuda":
                    peak_memory = torch.cuda.max_memory_allocated()
                    current_memory = torch.cuda.memory_allocated()
                    memory_used = (current_memory - initial_memory) / 1024 / 1024  # MB
                    peak_memory_mb = peak_memory / 1024 / 1024  # MB
                else:
                    process = psutil.Process(os.getpid())
                    memory_used = process.memory_info().rss / 1024 / 1024  # MB
                    peak_memory_mb = memory_used
                
                result = {
                    'seq_len': seq_len,
                    'memory_used_mb': memory_used,
                    'peak_memory_mb': peak_memory_mb,
                    'device': device
                }
                
                memory_results[f"seq{seq_len}"] = result
                
                print(f"     ✅ Memory used: {memory_used:.1f} MB (peak: {peak_memory_mb:.1f} MB)")
                
            except Exception as e:
                print(f"     ❌ Failed: {e}")
                continue
        
        self.results['memory'] = memory_results
        return memory_results
    
    def benchmark_generation(self, model, prompt="The future of AI is", max_length=50):
        """Benchmark text generation"""
        print(f"\n📝 Benchmarking Text Generation...")
        print(f"   Prompt: '{prompt}'")
        print(f"   Max length: {max_length}")
        
        if model is None:
            print("❌ No model to benchmark")
            return {}
            
        try:
            # Simple tokenization (just use character-level for demo)
            input_text = prompt
            input_ids = torch.tensor([[ord(c) % 1000 for c in input_text[:20]]])  # Simple encoding
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if device == "cuda":
                model = model.cuda()
                input_ids = input_ids.cuda()
            
            start_time = time.time()
            
            # Simple generation loop
            generated_ids = input_ids.clone()
            
            for _ in range(max_length - input_ids.shape[1]):
                with torch.no_grad():
                    outputs = model(generated_ids)
                    logits = outputs.logits
                    
                    # Get next token (greedy)
                    next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                    generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            end_time = time.time()
            
            generation_time = end_time - start_time
            tokens_generated = generated_ids.shape[1] - input_ids.shape[1]
            generation_speed = tokens_generated / generation_time
            
            result = {
                'prompt_length': input_ids.shape[1],
                'tokens_generated': tokens_generated,
                'generation_time_s': generation_time,
                'generation_speed_tok_per_sec': generation_speed,
                'device': device
            }
            
            print(f"     ✅ Generated {tokens_generated} tokens in {generation_time:.2f}s")
            print(f"     ⚡ Generation speed: {generation_speed:.1f} tok/sec")
            
            self.results['generation'] = result
            return result
            
        except Exception as e:
            print(f"     ❌ Generation failed: {e}")
            return {}
    
    def run_comprehensive_benchmark(self):
        """Run all benchmarks"""
        print("🧪 Running Comprehensive Hymba Performance Benchmark")
        print("=" * 60)
        
        # Create model
        model = self.create_model()
        if model is None:
            print("❌ Cannot proceed without model")
            return {}
        
        # Run benchmarks
        throughput_results = self.benchmark_throughput(model)
        memory_results = self.benchmark_memory_usage(model)
        generation_results = self.benchmark_generation(model)
        
        # Generate summary
        self.generate_summary_report()
        
        return self.results
    
    def generate_summary_report(self):
        """Generate summary report"""
        print("\n" + "=" * 60)
        print("📊 Performance Benchmark Summary")
        print("=" * 60)
        
        # Throughput summary
        if 'throughput' in self.results:
            print("\n⚡ Throughput Results:")
            for key, result in self.results['throughput'].items():
                print(f"   {key}: {result['throughput_tok_per_sec']:.1f} tok/sec ({result['avg_time_ms']:.1f}ms)")
        
        # Memory summary
        if 'memory' in self.results:
            print("\n💾 Memory Usage:")
            for key, result in self.results['memory'].items():
                print(f"   {key}: {result['memory_used_mb']:.1f} MB")
        
        # Generation summary
        if 'generation' in self.results:
            result = self.results['generation']
            print(f"\n📝 Generation: {result['generation_speed_tok_per_sec']:.1f} tok/sec")
        
        # Compare with paper claims
        print("\n📋 Comparison with Paper Claims:")
        print("   Paper claims (Hymba-1.5B vs Llama-3.2-3B):")
        print("   - Throughput: 664 vs 191 tok/sec (3.48x faster)")
        print("   - Cache size: 79MB vs 918MB (11.67x smaller)")
        
        if 'throughput' in self.results:
            best_throughput = max(r['throughput_tok_per_sec'] for r in self.results['throughput'].values())
            print(f"   - Our throughput: {best_throughput:.1f} tok/sec")
        
        if 'memory' in self.results:
            min_memory = min(r['memory_used_mb'] for r in self.results['memory'].values())
            print(f"   - Our memory: {min_memory:.1f} MB")


def main():
    """Main benchmark script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hymba Performance Benchmark")
    parser.add_argument("--model", help="Model path (optional)")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark only")
    
    args = parser.parse_args()
    
    benchmark = HymbaPerformanceBenchmark(args.model)
    
    if args.quick:
        # Quick benchmark
        model = benchmark.create_model()
        if model:
            benchmark.benchmark_throughput(model, seq_lengths=[128], batch_sizes=[1])
            benchmark.generate_summary_report()
    else:
        # Full benchmark
        benchmark.run_comprehensive_benchmark()


if __name__ == "__main__":
    main()
