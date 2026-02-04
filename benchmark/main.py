#!/usr/bin/env python3
"""
LLM Firewall Benchmark - Main Runner
Orchestrates evaluation of multiple firewalls against curated datasets
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add current directory to path
sys.path.insert(0, "/app")

from firewalls.rebuff_fw import RebuffFirewall
from firewalls.promptguard_fw import PromptGuardFirewall
from firewalls.nemo_fw import NeMoGuardrailsFirewall
from firewalls.trylon_fw import TrylonFirewall
from firewalls.llamaguard_fw import LlamaGuardFirewall
from firewalls.base import Firewall

from evaluation.metrics import MetricsCalculator, FirewallMetrics
from evaluation.logger import BenchmarkLogger, ConsoleLogger


class BenchmarkRunner:
    """Main benchmark orchestrator"""
    
    def __init__(self, datasets_dir: str = "/app/datasets", results_dir: str = "/app/results"):
        self.datasets_dir = Path(datasets_dir)
        self.results_dir = Path(results_dir)
        
        self.logger = BenchmarkLogger(output_dir=str(self.results_dir))
        self.console = ConsoleLogger()
        self.metrics_calc = MetricsCalculator()
        
        self.firewalls: List[Firewall] = []
        self.datasets: Dict[str, List[Dict]] = {}
    
    def load_datasets(self, use_hf_datasets: bool = False, use_combined: bool = False):
        """Load all JSON datasets"""
        self.console.header("LOADING DATASETS")
        
        if use_combined:
            # Use combined datasets from data/ subdirectory
            dataset_files = [
                "data/combined_injection.json",
                "data/combined_jailbreak.json",
                "data/combined_benign.json"
            ]
            self.console.info("Using COMBINED datasets (HuggingFace + Kaggle)")
        elif use_hf_datasets:
            dataset_files = [
                "prompt_injection_hf.json",
                "jailbreak_hf.json",
                "safe_prompts_hf.json"
            ]
            self.console.info("Using HuggingFace datasets")
        else:
            dataset_files = [
                "prompt_injection.json",
                "jailbreak.json",
                "safe_prompts.json"
            ]
            self.console.info("Using manually curated datasets")
        
        total_prompts = 0
        for filename in dataset_files:
            filepath = self.datasets_dir / filename
            if filepath.exists():
                with open(filepath, "r") as f:
                    data = json.load(f)
                    
                    # Normalize labels for combined/HuggingFace datasets
                    if use_combined or use_hf_datasets:
                        for item in data:
                            label = item.get("label", "")
                            if label in ["jailbreak", "injection"]:
                                item["label"] = "attack"
                            elif label == "safe":
                                item["label"] = "safe"
                    
                    # Normalize filename for compatibility
                    key = filename.replace("_hf", "").replace("combined_", "").replace("data/", "")
                    if key == "benign.json":
                        key = "safe_prompts.json"
                    self.datasets[key] = data
                    total_prompts += len(data)
                    self.console.success(f"Loaded {filename}: {len(data)} prompts")
            else:
                self.console.warning(f"Dataset not found: {filepath}")
        
        self.console.info(f"Total prompts loaded: {total_prompts}")
        return total_prompts > 0
    
    def initialize_firewalls(self, selected_firewalls: List[str] = None):
        """Initialize selected firewalls"""
        self.console.header("INITIALIZING FIREWALLS")
        
        # Define all available firewalls
        available_firewalls = {
            "rebuff": RebuffFirewall(),
            "promptguard": PromptGuardFirewall(),
            "nemo": NeMoGuardrailsFirewall(),
            "trylon": TrylonFirewall(),
            "llamaguard": LlamaGuardFirewall()
        }
        
        # Use selected or all firewalls
        if selected_firewalls is None:
            selected_firewalls = list(available_firewalls.keys())
        
        for name in selected_firewalls:
            if name in available_firewalls:
                firewall = available_firewalls[name]
                self.console.info(f"Initializing {firewall.name}...")
                
                if firewall.initialize():
                    self.firewalls.append(firewall)
                else:
                    self.console.warning(f"Failed to initialize {firewall.name} - skipping")
            else:
                self.console.warning(f"Unknown firewall: {name}")
        
        self.console.success(f"Initialized {len(self.firewalls)} firewall(s)")
        return len(self.firewalls) > 0
    
    def run_evaluation(self):
        """Run benchmark evaluation"""
        self.console.header("RUNNING BENCHMARK")
        
        all_metrics: List[FirewallMetrics] = []
        
        # Evaluate each firewall
        for firewall in self.firewalls:
            self.console.section(f"Evaluating: {firewall.name}")
            
            predictions = []
            ground_truths = []
            latencies = []
            results_data = []
            
            # Combine all datasets
            all_prompts = []
            for dataset_name, prompts in self.datasets.items():
                all_prompts.extend(prompts)
            
            total = len(all_prompts)
            
            # Evaluate each prompt
            for idx, prompt_data in enumerate(all_prompts, 1):
                # Handle datasets with or without 'id' field
                prompt_id = prompt_data.get("id", f"prompt_{idx}")
                prompt_text = prompt_data["prompt"]
                ground_truth = prompt_data["label"]
                
                # Show progress
                self.console.progress(idx, total, prefix=f"{firewall.name}")
                
                # Evaluate
                try:
                    result = firewall.evaluate(prompt_text)
                    
                    # Log result
                    self.logger.log_results(
                        firewall_name=firewall.name,
                        prompt_id=prompt_id,
                        prompt=prompt_text,
                        ground_truth=ground_truth,
                        result=result
                    )
                    
                    # Collect for metrics
                    predictions.append(result.decision)
                    ground_truths.append(ground_truth)
                    latencies.append(result.latency_ms)
                    
                    # Determine if prediction was correct
                    expected_decision = "BLOCK" if ground_truth == "attack" else "ALLOW"
                    correct = result.decision == expected_decision
                    
                    results_data.append({
                        "prompt_id": prompt_id,
                        "ground_truth": ground_truth,
                        "decision": result.decision,
                        "confidence": result.confidence,
                        "latency_ms": result.latency_ms,
                        "correct": correct
                    })
                    
                except Exception as e:
                    self.console.error(f"Error evaluating prompt {prompt_id}: {e}")
                    predictions.append("ERROR")
                    ground_truths.append(ground_truth)
                    latencies.append(0.0)
            
            print()  # New line after progress bar
            
            # Calculate metrics
            metrics = self.metrics_calc.calculate_metrics(
                firewall_name=firewall.name,
                predictions=predictions,
                ground_truth=ground_truths,
                latencies=latencies
            )
            
            all_metrics.append(metrics)
            
            # Save individual results
            self.logger.save_csv_summary(firewall.name, results_data)
            
            # Generate and save report
            report = self.metrics_calc.generate_report(metrics)
            print(report)
            self.logger.save_metrics_report(report, f"{firewall.name}_metrics.txt")
            
            # Cleanup firewall
            firewall.cleanup()
        
        # Generate comparison
        if len(all_metrics) > 1:
            self.console.section("FIREWALL COMPARISON")
            comparison = self.metrics_calc.compare_firewalls(all_metrics)
            print(comparison)
            self.logger.save_comparison(comparison)
        
        # Save summary
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_firewalls": len(all_metrics),
            "total_prompts": len(all_prompts),
            "datasets": list(self.datasets.keys()),
            "firewalls": [m.firewall_name for m in all_metrics],
            "metrics": [
                {
                    "firewall": m.firewall_name,
                    "accuracy": m.accuracy,
                    "precision": m.precision,
                    "recall": m.recall,
                    "f1_score": m.f1_score,
                    "avg_latency_ms": m.avg_latency_ms,
                    "attack_success_rate": m.attack_success_rate
                }
                for m in all_metrics
            ]
        }
        self.logger.save_summary_json(summary)
        
        self.console.success(f"✅ Benchmark complete! Results saved to: {self.logger.get_results_path()}")


def main():
    """Main entry point"""
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="LLM Firewall Benchmark")
    parser.add_argument(
        "--firewalls",
        nargs="+",
        choices=["rebuff", "promptguard", "nemo", "trylon", "llamaguard"],
        help="Firewalls to test (default: all)"
    )
    parser.add_argument(
        "--datasets-dir",
        default="/app/datasets",
        help="Path to datasets directory"
    )
    parser.add_argument(
        "--results-dir",
        default="/app/results",
        help="Path to results directory"
    )
    parser.add_argument(
        "--use-hf-datasets",
        action="store_true",
        help="Use HuggingFace datasets instead of manually curated ones"
    )
    parser.add_argument(
        "--use-combined",
        action="store_true",
        help="Use combined datasets (HuggingFace + Kaggle, 10K+ prompts)"
    )
    
    args = parser.parse_args()
    
    # Create and run benchmark
    runner = BenchmarkRunner(
        datasets_dir=args.datasets_dir,
        results_dir=args.results_dir
    )
    
    # Load datasets
    if not runner.load_datasets(use_hf_datasets=args.use_hf_datasets, use_combined=args.use_combined):
        print("❌ Failed to load datasets. Exiting.")
        sys.exit(1)
    
    # Initialize firewalls
    if not runner.initialize_firewalls(selected_firewalls=args.firewalls):
        print("❌ Failed to initialize any firewalls. Exiting.")
        sys.exit(1)
    
    # Run evaluation
    try:
        runner.run_evaluation()
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()