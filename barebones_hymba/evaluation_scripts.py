#!/usr/bin/env python3
"""
Evaluation scripts for Hymba model using lm-evaluation-harness
"""

import subprocess
import json
import os
import sys
import time
from pathlib import Path


class HymbaEvaluator:
    """Evaluation suite for Hymba model"""
    
    def __init__(self, model_path=None, output_dir="evaluation_results"):
        import os
        # Use absolute path for local model
        if model_path is None:
            self.model_path = os.path.abspath(".")
        else:
            self.model_path = model_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Tasks as specified in the paper
        self.tasks = {
            'mmlu': {
                'name': 'mmlu',
                'num_fewshot': 5,
                'description': 'Massive Multitask Language Understanding'
            },
            'arc_easy': {
                'name': 'arc_easy',
                'num_fewshot': 0,
                'description': 'AI2 Reasoning Challenge (Easy)'
            },
            'arc_challenge': {
                'name': 'arc_challenge', 
                'num_fewshot': 0,
                'description': 'AI2 Reasoning Challenge (Challenge)'
            },
            'piqa': {
                'name': 'piqa',
                'num_fewshot': 0,
                'description': 'Physical Interaction QA'
            },
            'hellaswag': {
                'name': 'hellaswag',
                'num_fewshot': 0,
                'description': 'HellaSwag Commonsense Reasoning'
            },
            'winogrande': {
                'name': 'winogrande',
                'num_fewshot': 0,
                'description': 'Winograd Schema Challenge'
            },
            'squad_completion': {
                'name': 'squad_completion',
                'num_fewshot': 1,
                'description': 'SQuAD Reading Comprehension'
            }
        }
        
        # Expected results from paper (for comparison)
        self.paper_results = {
            'mmlu': 51.19,
            'arc_easy': 76.94,
            'arc_challenge': 45.90,
            'piqa': 77.31,
            'hellaswag': 53.55,
            'winogrande': 66.61,
            'squad_completion': 55.93
        }
    
    def run_single_task(self, task_name, batch_size=1, device="auto"):
        """Run evaluation for a single task"""
        
        if task_name not in self.tasks:
            raise ValueError(f"Unknown task: {task_name}")
            
        task_config = self.tasks[task_name]
        output_file = self.output_dir / f"{task_name}_results.json"
        
        print(f"\n🔄 Running {task_config['description']} ({task_name})...")
        print(f"   Few-shot: {task_config['num_fewshot']}")
        print(f"   Output: {output_file}")
        
        # Build lm_eval command
        cmd = [
            "lm_eval",
            "--model", "hf",
            "--model_args", f"pretrained={self.model_path},dtype=bfloat16,trust_remote_code=True",
            "--tasks", task_config['name'],
            "--num_fewshot", str(task_config['num_fewshot']),
            "--batch_size", str(batch_size),
            "--device", device,
            "--output_path", str(output_file)
        ]
        
        try:
            start_time = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)  # 1 hour timeout
            end_time = time.time()
            
            if result.returncode == 0:
                print(f"✅ {task_name} completed in {end_time - start_time:.1f}s")
                return self._parse_results(output_file, task_name)
            else:
                print(f"❌ {task_name} failed:")
                print(f"   stdout: {result.stdout}")
                print(f"   stderr: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {task_name} timed out after 1 hour")
            return None
        except Exception as e:
            print(f"❌ {task_name} error: {e}")
            return None
    
    def _parse_results(self, output_file, task_name):
        """Parse evaluation results from output file"""
        try:
            if output_file.exists():
                with open(output_file, 'r') as f:
                    data = json.load(f)
                    
                # Extract main metric (usually accuracy)
                results = data.get('results', {})
                task_results = results.get(task_name, {})
                
                # Common metrics to look for
                metrics = ['acc', 'acc_norm', 'exact_match', 'f1']
                score = None
                metric_name = None
                
                for metric in metrics:
                    if metric in task_results:
                        score = task_results[metric]
                        metric_name = metric
                        break
                
                return {
                    'task': task_name,
                    'score': score,
                    'metric': metric_name,
                    'raw_results': task_results
                }
            else:
                print(f"⚠️  Output file not found: {output_file}")
                return None
                
        except Exception as e:
            print(f"❌ Error parsing results for {task_name}: {e}")
            return None
    
    def run_all_tasks(self, batch_size=1, device="auto"):
        """Run evaluation for all tasks"""
        
        print("🧪 Running Complete Hymba Evaluation Suite")
        print("=" * 60)
        print(f"Model: {self.model_path}")
        print(f"Output directory: {self.output_dir}")
        print(f"Batch size: {batch_size}")
        print(f"Device: {device}")
        print("=" * 60)
        
        results = {}
        total_tasks = len(self.tasks)
        completed_tasks = 0
        
        for task_name in self.tasks:
            result = self.run_single_task(task_name, batch_size, device)
            if result:
                results[task_name] = result
                completed_tasks += 1
            
            print(f"Progress: {completed_tasks}/{total_tasks} tasks completed")
        
        # Generate summary report
        self._generate_summary_report(results)
        
        return results
    
    def _generate_summary_report(self, results):
        """Generate summary evaluation report"""
        
        summary_file = self.output_dir / "evaluation_summary.json"
        report_file = self.output_dir / "evaluation_report.txt"
        
        # Save detailed results
        with open(summary_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate text report
        with open(report_file, 'w') as f:
            f.write("Hymba Model Evaluation Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Task Results:\n")
            f.write("-" * 30 + "\n")
            
            total_score = 0
            valid_tasks = 0
            
            for task_name, result in results.items():
                if result and result['score'] is not None:
                    score = result['score'] * 100  # Convert to percentage
                    paper_score = self.paper_results.get(task_name, 0)
                    diff = score - paper_score
                    
                    f.write(f"{task_name:<15} | {score:>6.2f}% | Paper: {paper_score:>6.2f}% | Diff: {diff:>+6.2f}%\n")
                    
                    total_score += score
                    valid_tasks += 1
                else:
                    f.write(f"{task_name:<15} | {'FAILED':>6} | Paper: {self.paper_results.get(task_name, 0):>6.2f}% | Diff: {'N/A':>6}\n")
            
            f.write("-" * 30 + "\n")
            
            if valid_tasks > 0:
                avg_score = total_score / valid_tasks
                f.write(f"{'Average':<15} | {avg_score:>6.2f}% | Completed: {valid_tasks}/{len(self.tasks)} tasks\n")
            
            f.write("\nTask Descriptions:\n")
            f.write("-" * 30 + "\n")
            for task_name, config in self.tasks.items():
                f.write(f"{task_name}: {config['description']} ({config['num_fewshot']}-shot)\n")
        
        print(f"\n📊 Summary saved to: {summary_file}")
        print(f"📋 Report saved to: {report_file}")
    
    def run_quick_test(self, task_name="piqa", batch_size=1):
        """Run a quick test on a single task"""
        print(f"🚀 Running quick test on {task_name}...")
        result = self.run_single_task(task_name, batch_size)
        
        if result:
            score = result['score'] * 100 if result['score'] else 0
            paper_score = self.paper_results.get(task_name, 0)
            print(f"\n📊 Quick Test Results:")
            print(f"   Task: {task_name}")
            print(f"   Score: {score:.2f}%")
            print(f"   Paper: {paper_score:.2f}%")
            print(f"   Difference: {score - paper_score:+.2f}%")
        
        return result


def main():
    """Main evaluation script"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hymba Model Evaluation")
    parser.add_argument("--model", default="barebones_hymba", help="Model path")
    parser.add_argument("--output", default="evaluation_results", help="Output directory")
    parser.add_argument("--task", help="Single task to run (default: run all)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--device", default="auto", help="Device to use")
    parser.add_argument("--quick", action="store_true", help="Run quick test only")
    
    args = parser.parse_args()
    
    evaluator = HymbaEvaluator(args.model, args.output)
    
    if args.quick:
        evaluator.run_quick_test()
    elif args.task:
        evaluator.run_single_task(args.task, args.batch_size, args.device)
    else:
        evaluator.run_all_tasks(args.batch_size, args.device)


if __name__ == "__main__":
    main()
