"""
Flask-based Web UI for LLM Firewall Benchmark Results
"""
import os
import json
import glob
from flask import Flask, render_template, jsonify, send_from_directory
from datetime import datetime
import csv

app = Flask(__name__)
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')

def get_all_runs():
    """Get all benchmark runs from results directory"""
    runs = []
    
    if not os.path.exists(RESULTS_DIR):
        return runs
    
    for dataset_dir in os.listdir(RESULTS_DIR):
        dataset_path = os.path.join(RESULTS_DIR, dataset_dir)
        if not os.path.isdir(dataset_path):
            continue
            
        for run_dir in os.listdir(dataset_path):
            run_path = os.path.join(dataset_path, run_dir)
            if os.path.isdir(run_path) and run_dir.startswith('run_'):
                runs.append({
                    'dataset': dataset_dir,
                    'run_id': run_dir,
                    'path': run_path,
                    'timestamp': run_dir.replace('run_', '')
                })
    
    # Sort by timestamp descending (newest first)
    runs.sort(key=lambda x: x['timestamp'], reverse=True)
    return runs

def parse_metrics_file(filepath):
    """Parse a firewall metrics file"""
    if not os.path.exists(filepath):
        return None
    
    metrics = {}
    with open(filepath, 'r') as f:
        content = f.read()
        
        # Parse metrics using simple string parsing
        for line in content.split('\n'):
            line = line.strip()
            if 'Accuracy:' in line:
                metrics['accuracy'] = float(line.split('(')[1].split('%')[0])
            elif 'Precision:' in line:
                metrics['precision'] = float(line.split('(')[1].split('%')[0])
            elif 'Recall:' in line:
                metrics['recall'] = float(line.split('(')[1].split('%')[0])
            elif 'F1 Score:' in line:
                metrics['f1'] = float(line.split(':')[1].strip())
            elif 'True Positives' in line:
                metrics['tp'] = int(line.split(':')[1].strip())
            elif 'True Negatives' in line:
                metrics['tn'] = int(line.split(':')[1].strip())
            elif 'False Positives' in line:
                metrics['fp'] = int(line.split(':')[1].strip())
            elif 'False Negatives' in line:
                metrics['fn'] = int(line.split(':')[1].strip())
            elif 'Average Latency:' in line:
                latency_str = line.split(':')[1].strip().split()[0]
                metrics['latency'] = float(latency_str)
            elif 'Total Prompts:' in line:
                metrics['total_prompts'] = int(line.split(':')[1].strip())
            elif 'Errors:' in line:
                metrics['errors'] = int(line.split(':')[1].strip())
            elif 'Attack Success Rate:' in line:
                metrics['asr'] = float(line.split(':')[1].strip().split('%')[0])
    
    return metrics

def get_run_results(run_path):
    """Get all firewall results for a specific run"""
    firewalls = ['Rebuff', 'PromptGuard', 'NeMo-Guardrails', 'Trylon', 'LlamaGuard']
    results = {}
    
    for firewall in firewalls:
        metrics_file = os.path.join(run_path, f"{firewall}_metrics.txt")
        csv_file = os.path.join(run_path, f"{firewall}_summary.csv")
        
        if os.path.exists(metrics_file):
            metrics = parse_metrics_file(metrics_file)
            if metrics:
                results[firewall] = {
                    'metrics': metrics,
                    'has_csv': os.path.exists(csv_file),
                    'csv_file': f"{firewall}_summary.csv" if os.path.exists(csv_file) else None
                }
    
    return results

def get_comparison_data(run_path):
    """Get comparison data from firewall_comparison.txt"""
    comparison_file = os.path.join(run_path, 'firewall_comparison.txt')
    if os.path.exists(comparison_file):
        with open(comparison_file, 'r') as f:
            return f.read()
    return None

@app.route('/')
def index():
    """Main dashboard page"""
    runs = get_all_runs()
    latest_run = runs[0] if runs else None
    
    if latest_run:
        results = get_run_results(latest_run['path'])
        comparison = get_comparison_data(latest_run['path'])
    else:
        results = {}
        comparison = None
    
    return render_template('index.html', 
                         runs=runs, 
                         latest_run=latest_run,
                         results=results,
                         comparison=comparison)

@app.route('/api/runs')
def api_runs():
    """API endpoint for all runs"""
    runs = get_all_runs()
    return jsonify(runs)

@app.route('/api/run/<dataset>/<run_id>')
def api_run_details(dataset, run_id):
    """API endpoint for specific run details"""
    run_path = os.path.join(RESULTS_DIR, dataset, run_id)
    
    if not os.path.exists(run_path):
        return jsonify({'error': 'Run not found'}), 404
    
    results = get_run_results(run_path)
    comparison = get_comparison_data(run_path)
    
    return jsonify({
        'dataset': dataset,
        'run_id': run_id,
        'results': results,
        'comparison': comparison
    })

@app.route('/api/progress')
def api_progress():
    """API endpoint for real-time benchmark progress"""
    runs = get_all_runs()
    if not runs:
        return jsonify({'status': 'no_runs', 'progress': 0})
    
    latest_run = runs[0]
    results = get_run_results(latest_run['path'])
    
    # Count completed firewalls
    firewalls = ['Rebuff', 'PromptGuard', 'NeMo-Guardrails', 'Trylon', 'LlamaGuard']
    completed = len(results)
    total = len(firewalls)
    
    return jsonify({
        'status': 'complete' if completed == total else 'running',
        'progress': (completed / total) * 100,
        'completed': completed,
        'total': total,
        'firewalls': list(results.keys())
    })

@app.route('/download/<dataset>/<run_id>/<filename>')
def download_file(dataset, run_id, filename):
    """Download CSV or results files"""
    run_path = os.path.join(RESULTS_DIR, dataset, run_id)
    return send_from_directory(run_path, filename, as_attachment=True)

@app.route('/view/<dataset>/<run_id>')
def view_run(dataset, run_id):
    """View specific run details"""
    run_path = os.path.join(RESULTS_DIR, dataset, run_id)
    
    if not os.path.exists(run_path):
        return "Run not found", 404
    
    results = get_run_results(run_path)
    comparison = get_comparison_data(run_path)
    
    return render_template('run_detail.html',
                         dataset=dataset,
                         run_id=run_id,
                         results=results,
                         comparison=comparison)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
