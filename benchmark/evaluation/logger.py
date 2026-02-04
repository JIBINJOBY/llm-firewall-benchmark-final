"""
Logger Module
Handles logging of benchmark results to files and console
"""

import json
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
from firewalls.base import FirewallResult


class BenchmarkLogger:
    """
    Logs benchmark results in multiple formats (JSON, CSV, TXT)
    """
    
    def __init__(self, output_dir: str = "/app/results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate timestamp for this run
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"run_{self.timestamp}"
        
        # Create subdirectory for this run
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"📁 Results will be saved to: {self.run_dir}")
    
    def log_results(
        self,
        firewall_name: str,
        prompt_id: int,
        prompt: str,
        ground_truth: str,
        result: FirewallResult
    ):
        """Log a single evaluation result"""
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "firewall": firewall_name,
            "prompt_id": prompt_id,
            "prompt": prompt,
            "ground_truth": ground_truth,
            "decision": result.decision,
            "confidence": result.confidence,
            "latency_ms": result.latency_ms,
            "metadata": result.metadata
        }
        
        # Append to JSON lines file
        jsonl_path = self.run_dir / f"{firewall_name}_results.jsonl"
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def save_csv_summary(
        self,
        firewall_name: str,
        results: List[Dict[str, Any]]
    ):
        """Save results summary as CSV"""
        
        csv_path = self.run_dir / f"{firewall_name}_summary.csv"
        
        if not results:
            return
        
        # Define CSV columns
        fieldnames = [
            "prompt_id",
            "ground_truth",
            "decision",
            "confidence",
            "latency_ms",
            "correct"
        ]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for row in results:
                writer.writerow({
                    "prompt_id": row["prompt_id"],
                    "ground_truth": row["ground_truth"],
                    "decision": row["decision"],
                    "confidence": row["confidence"],
                    "latency_ms": row["latency_ms"],
                    "correct": row["correct"]
                })
        
        print(f"📊 CSV saved: {csv_path}")
    
    def save_metrics_report(self, report: str, filename: str = "metrics_report.txt"):
        """Save metrics report to text file"""
        
        report_path = self.run_dir / filename
        with open(report_path, "w") as f:
            f.write(report)
        
        print(f"📄 Report saved: {report_path}")
    
    def save_comparison(self, comparison: str):
        """Save firewall comparison to file"""
        
        comparison_path = self.run_dir / "firewall_comparison.txt"
        with open(comparison_path, "w") as f:
            f.write(comparison)
        
        print(f"📊 Comparison saved: {comparison_path}")
    
    def save_summary_json(self, summary: Dict[str, Any]):
        """Save overall summary as JSON"""
        
        summary_path = self.run_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Summary JSON saved: {summary_path}")
    
    def get_results_path(self) -> Path:
        """Get the path to results directory"""
        return self.run_dir


class ConsoleLogger:
    """Simple console logger with formatting"""
    
    @staticmethod
    def header(text: str):
        """Print a header"""
        print(f"\n{'='*80}")
        print(f"  {text}")
        print(f"{'='*80}\n")
    
    @staticmethod
    def section(text: str):
        """Print a section header"""
        print(f"\n{'-'*80}")
        print(f"  {text}")
        print(f"{'-'*80}")
    
    @staticmethod
    def info(text: str):
        """Print info message"""
        print(f"ℹ️  {text}")
    
    @staticmethod
    def success(text: str):
        """Print success message"""
        print(f"✅ {text}")
    
    @staticmethod
    def warning(text: str):
        """Print warning message"""
        print(f"⚠️  {text}")
    
    @staticmethod
    def error(text: str):
        """Print error message"""
        print(f"❌ {text}")
    
    @staticmethod
    def progress(current: int, total: int, prefix: str = "Progress"):
        """Print progress"""
        percentage = (current / total) * 100 if total > 0 else 0
        bar_length = 40
        filled = int(bar_length * current / total) if total > 0 else 0
        bar = '█' * filled + '-' * (bar_length - filled)
        print(f"\r{prefix}: |{bar}| {current}/{total} ({percentage:.1f}%)", end="", flush=True)
        if current == total:
            print()  # New line when complete
