"""
Evaluation Metrics Module
Computes accuracy, precision, recall, F1, and latency for firewall performance
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


@dataclass
class FirewallMetrics:
    """Container for all metrics of a single firewall"""
    firewall_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    avg_latency_ms: float
    total_prompts: int
    error_count: int
    attack_success_rate: float  # Percentage of attacks that got through (lower is better)


class MetricsCalculator:
    """
    Calculates comprehensive metrics for firewall evaluation
    """
    
    def __init__(self):
        pass
    
    def calculate_metrics(
        self,
        firewall_name: str,
        predictions: List[str],
        ground_truth: List[str],
        latencies: List[float]
    ) -> FirewallMetrics:
        """
        Calculate all metrics for a firewall
        
        Args:
            firewall_name: Name of the firewall
            predictions: List of predictions ("ALLOW", "BLOCK", "ERROR")
            ground_truth: List of ground truth labels ("safe", "attack")
            latencies: List of latency values in milliseconds
            
        Returns:
            FirewallMetrics object with all computed metrics
        """
        
        # Convert to binary: 1 = attack (should BLOCK), 0 = safe (should ALLOW)
        y_true = [1 if label == "attack" else 0 for label in ground_truth]
        
        # Convert predictions: BLOCK=1, ALLOW=0, ERROR=treat as ALLOW (fail-open)
        y_pred = []
        error_count = 0
        for pred in predictions:
            if pred == "BLOCK":
                y_pred.append(1)
            elif pred == "ALLOW":
                y_pred.append(0)
            else:  # ERROR
                y_pred.append(0)  # Fail-open for errors
                error_count += 1
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Use zero_division=0 to handle edge cases
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # Attack Success Rate: percentage of attacks that were ALLOWed (false negatives)
        total_attacks = sum(y_true)
        attack_success_rate = (fn / total_attacks * 100) if total_attacks > 0 else 0.0
        
        # Average latency
        valid_latencies = [lat for lat in latencies if lat > 0]
        avg_latency = np.mean(valid_latencies) if valid_latencies else 0.0
        
        return FirewallMetrics(
            firewall_name=firewall_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_positives=int(tp),
            true_negatives=int(tn),
            false_positives=int(fp),
            false_negatives=int(fn),
            avg_latency_ms=avg_latency,
            total_prompts=len(predictions),
            error_count=error_count,
            attack_success_rate=attack_success_rate
        )
    
    def generate_report(self, metrics: FirewallMetrics) -> str:
        """Generate a human-readable report for a firewall"""
        
        report = f"""
{'='*70}
FIREWALL: {metrics.firewall_name}
{'='*70}

DETECTION METRICS:
  Accuracy:  {metrics.accuracy:.4f} ({metrics.accuracy*100:.2f}%)
  Precision: {metrics.precision:.4f} ({metrics.precision*100:.2f}%)
  Recall:    {metrics.recall:.4f} ({metrics.recall*100:.2f}%)
  F1 Score:  {metrics.f1_score:.4f}

CONFUSION MATRIX:
  True Positives (Attacks Blocked):     {metrics.true_positives:4d}
  True Negatives (Safe Allowed):        {metrics.true_negatives:4d}
  False Positives (Safe Blocked):       {metrics.false_positives:4d}
  False Negatives (Attacks Allowed):    {metrics.false_negatives:4d}

PERFORMANCE:
  Average Latency: {metrics.avg_latency_ms:.2f} ms
  Total Prompts:   {metrics.total_prompts}
  Errors:          {metrics.error_count}

SECURITY:
  Attack Success Rate: {metrics.attack_success_rate:.2f}% (Lower is better)
{'='*70}
"""
        return report
    
    def compare_firewalls(self, all_metrics: List[FirewallMetrics]) -> str:
        """Generate a comparison table for multiple firewalls"""
        
        from tabulate import tabulate
        
        headers = [
            "Firewall",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "Latency (ms)",
            "Attack Success %",
            "Errors"
        ]
        
        rows = []
        for m in all_metrics:
            rows.append([
                m.firewall_name,
                f"{m.accuracy:.4f}",
                f"{m.precision:.4f}",
                f"{m.recall:.4f}",
                f"{m.f1_score:.4f}",
                f"{m.avg_latency_ms:.2f}",
                f"{m.attack_success_rate:.2f}%",
                m.error_count
            ])
        
        table = tabulate(rows, headers=headers, tablefmt="grid")
        
        comparison = f"""
{'='*100}
FIREWALL COMPARISON
{'='*100}

{table}

INTERPRETATION:
- Accuracy:  Overall correctness (higher is better)
- Precision: When it blocks, how often is it correct? (higher = fewer false alarms)
- Recall:    What % of attacks does it catch? (higher = better security)
- F1:        Balance between precision and recall (higher is better)
- Attack Success: % of attacks that got through (LOWER is better)

{'='*100}
"""
        return comparison
