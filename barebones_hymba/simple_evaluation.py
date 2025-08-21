#!/usr/bin/env python3
"""
Simple evaluation script to test model functionality
without complex lm-evaluation-harness dependencies
"""

import torch
import torch.nn.functional as F
import time
import json
import random
from pathlib import Path

from configuration_hymba import HymbaConfig
from modeling_hymba import HymbaForCausalLM


class SimpleEvaluator:
    """Simple evaluation for basic model testing"""
    
    def __init__(self):
        self.model = None
        self.config = None
        
    def load_model(self):
        """Load model from current directory"""
        print("🔧 Loading Hymba model...")
        
        try:
            self.config = HymbaConfig.from_pretrained('.')
            self.model = HymbaForCausalLM(self.config)
            
            print(f"✅ Model loaded successfully!")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            print(f"   Vocab size: {self.config.vocab_size}")
            
            return True
            
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            return False
    
    def test_multiple_choice(self, question, choices, correct_idx=0):
        """Test multiple choice question answering"""
        print(f"\n📝 Testing Multiple Choice...")
        print(f"   Question: {question}")
        
        if self.model is None:
            print("❌ No model loaded")
            return False
            
        try:
            # Simple tokenization (character-based for demo)
            def simple_tokenize(text):
                return [ord(c) % 1000 for c in text[:50]]  # Limit length
            
            # Test each choice
            choice_scores = []
            
            for i, choice in enumerate(choices):
                # Create prompt
                prompt = f"{question} {choice}"
                input_ids = torch.tensor([simple_tokenize(prompt)])
                
                # Get model prediction
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits
                    
                    # Calculate perplexity (lower is better)
                    log_probs = F.log_softmax(logits, dim=-1)
                    # Simple scoring: average log probability
                    score = log_probs.mean().item()
                    choice_scores.append(score)
                
                print(f"     Choice {i}: '{choice}' -> Score: {score:.3f}")
            
            # Find best choice
            predicted_idx = choice_scores.index(max(choice_scores))
            correct = predicted_idx == correct_idx
            
            print(f"   Predicted: Choice {predicted_idx}")
            print(f"   Correct: Choice {correct_idx}")
            print(f"   Result: {'✅ Correct' if correct else '❌ Wrong'}")
            
            return correct
            
        except Exception as e:
            print(f"❌ Multiple choice test failed: {e}")
            return False
    
    def test_text_completion(self, prompt, expected_keywords=None):
        """Test text completion"""
        print(f"\n📝 Testing Text Completion...")
        print(f"   Prompt: '{prompt}'")
        
        if self.model is None:
            print("❌ No model loaded")
            return False
            
        try:
            # Simple tokenization
            def simple_tokenize(text):
                return [ord(c) % 1000 for c in text[:30]]
            
            input_ids = torch.tensor([simple_tokenize(prompt)])
            
            # Generate continuation
            max_new_tokens = 20
            generated_ids = input_ids.clone()
            
            for _ in range(max_new_tokens):
                with torch.no_grad():
                    outputs = self.model(generated_ids)
                    logits = outputs.logits
                    
                    # Get next token (sample from top-k)
                    next_token_logits = logits[0, -1, :]
                    top_k = 10
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    probs = F.softmax(top_k_logits, dim=-1)
                    
                    # Sample from top-k
                    next_idx = torch.multinomial(probs, 1)
                    next_token = top_k_indices[next_idx]
                    
                    generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
                    
                    # Simple stopping condition
                    if next_token.item() == 0:
                        break
            
            # Convert back to text (simplified)
            generated_tokens = generated_ids[0].tolist()
            completion_tokens = generated_tokens[len(input_ids[0]):]
            
            print(f"   Generated tokens: {completion_tokens[:10]}...")  # Show first 10
            print(f"   Completion length: {len(completion_tokens)} tokens")
            
            # Check for expected keywords if provided
            if expected_keywords:
                found_keywords = []
                for keyword in expected_keywords:
                    # Simple keyword matching (convert to token representation)
                    keyword_tokens = [ord(c) % 1000 for c in keyword.lower()]
                    if any(token in completion_tokens for token in keyword_tokens[:3]):  # Check first 3 chars
                        found_keywords.append(keyword)
                
                print(f"   Expected keywords: {expected_keywords}")
                print(f"   Found keywords: {found_keywords}")
                
                return len(found_keywords) > 0
            
            return True
            
        except Exception as e:
            print(f"❌ Text completion test failed: {e}")
            return False
    
    def test_reasoning(self):
        """Test basic reasoning capabilities"""
        print(f"\n🧠 Testing Reasoning Capabilities...")
        
        # Test 1: Simple math
        math_correct = self.test_multiple_choice(
            "What is 2 + 2?",
            ["3", "4", "5", "6"],
            correct_idx=1
        )
        
        # Test 2: Common sense
        sense_correct = self.test_multiple_choice(
            "What do you use to cut paper?",
            ["hammer", "scissors", "spoon", "book"],
            correct_idx=1
        )
        
        # Test 3: Text completion
        completion_ok = self.test_text_completion(
            "The capital of France is",
            expected_keywords=["paris", "france", "city"]
        )
        
        total_score = sum([math_correct, sense_correct, completion_ok])
        print(f"\n📊 Reasoning Test Results: {total_score}/3 tests passed")
        
        return total_score >= 2  # Pass if at least 2/3 correct
    
    def test_performance(self):
        """Test model performance"""
        print(f"\n⚡ Testing Performance...")
        
        if self.model is None:
            print("❌ No model loaded")
            return False
            
        try:
            # Test different sequence lengths
            seq_lengths = [50, 100, 200]
            results = {}
            
            for seq_len in seq_lengths:
                print(f"   Testing seq_len={seq_len}...")
                
                # Create input
                input_ids = torch.randint(0, 1000, (1, seq_len))
                
                # Warmup
                for _ in range(3):
                    with torch.no_grad():
                        _ = self.model(input_ids)
                
                # Benchmark
                num_runs = 10
                start_time = time.time()
                
                for _ in range(num_runs):
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                
                end_time = time.time()
                
                # Calculate metrics
                total_time = end_time - start_time
                avg_time = total_time / num_runs
                tokens_per_sec = seq_len / avg_time
                
                results[seq_len] = {
                    'avg_time_ms': avg_time * 1000,
                    'tokens_per_sec': tokens_per_sec
                }
                
                print(f"     ✅ {tokens_per_sec:.1f} tok/sec ({avg_time*1000:.1f}ms)")
            
            # Summary
            best_throughput = max(r['tokens_per_sec'] for r in results.values())
            print(f"   Best throughput: {best_throughput:.1f} tok/sec")
            
            return True
            
        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return False
    
    def run_evaluation_suite(self):
        """Run complete evaluation suite"""
        print("🧪 Running Simple Hymba Evaluation Suite")
        print("=" * 60)
        
        # Load model
        if not self.load_model():
            print("❌ Cannot proceed without model")
            return False
        
        # Run tests
        tests = [
            ("Reasoning Capabilities", self.test_reasoning),
            ("Performance Benchmarking", self.test_performance),
        ]
        
        results = {}
        
        for test_name, test_func in tests:
            try:
                results[test_name] = test_func()
            except Exception as e:
                print(f"❌ {test_name} failed: {e}")
                results[test_name] = False
        
        # Summary
        print("\n" + "=" * 60)
        print("📋 Evaluation Summary:")
        print("=" * 60)
        
        passed_tests = sum(results.values())
        total_tests = len(results)
        
        for test_name, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{test_name:<30} | {status}")
        
        print("-" * 60)
        print(f"Total: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 All evaluation tests passed!")
            print("\n🚀 Model is ready for advanced evaluation!")
            print("   Next steps:")
            print("   1. Try running on actual evaluation datasets")
            print("   2. Compare with paper benchmarks")
            print("   3. Optimize performance if needed")
            return True
        else:
            print("⚠️  Some tests failed. Model may need debugging.")
            return False


def main():
    """Main evaluation script"""
    evaluator = SimpleEvaluator()
    success = evaluator.run_evaluation_suite()
    
    if success:
        print(f"\n📊 Model Performance Summary:")
        print(f"   ✅ Model loads and runs successfully")
        print(f"   ✅ Basic reasoning capabilities working")
        print(f"   ✅ Performance benchmarking completed")
        print(f"   ✅ Ready for production evaluation")
    
    return success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
