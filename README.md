# LLM Firewall Benchmark# 🛡️ LLM Firewall Benchmark Framework



Benchmarking framework for evaluating 5 LLM firewalls against prompt injection and jailbreak attacks.**A fully open-source, fully dockerized benchmark framework for evaluating LLM firewalls against prompt injection and safety attacks using local models.**



## Setup & Run[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://www.docker.com/)

[![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)](https://developer.nvidia.com/cuda-toolkit)

### 1. Clone & Configure[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)



```bash---

git clone <your-repo-url>

cd llm-firewall-benchmark## 🎯 What This Framework Does

```

Evaluates **5 LLM firewalls** against curated attack datasets:

Create `.env` file with API keys:

```bash### Firewalls Tested

REBUFF_API_KEY=your_rebuff_key- **Rebuff** - Fast heuristic + ML-based prompt injection detection

TRYLON_API_KEY=your_trylon_key- **PromptGuard** - Meta's Hugging Face model (86M parameters)

```- **NeMo Guardrails** - NVIDIA's rule-based guardrails

- **Trylon** - Gateway-based policy enforcement

### 2. Start Docker Services- **Llama Guard** - Meta's 8B safety classifier (powerful!)



```bash### Datasets

docker-compose up -d- **Prompt Injection** (15 attacks) - Instruction override attempts

```- **Jailbreak** (15 attacks) - Safety bypass attempts

- **Safe Prompts** (20 benign) - Legitimate queries

### 3. Download LLM Model

### Metrics Computed

```bash✅ Accuracy, Precision, Recall, F1 Score  

docker exec ollama ollama pull llama3.2:1b✅ Attack Success Rate (% of attacks that got through)  

```✅ Latency (ms per prompt)  

✅ Confusion Matrix  

### 4. Download Test Dataset

---

```bash

docker exec benchmark bash -c "cd /app && python3 -m datasets.load_hf_datasets --jailbreak 500 --injection 500 --benign 500"## 🏗️ Architecture

```

```

### 5. Run Benchmark┌─────────────────┐

│   Benchmark     │  (Python + GPU)

```bash│   Runner        │

docker exec benchmark python3 main.py --use-hf-datasets --results-dir results/benchmark_run└────────┬────────┘

```         │

    ┌────┴────┬──────────┬──────────┐

### 6. View Dashboard    │         │          │          │

    ▼         ▼          ▼          ▼

```bash Rebuff  PromptGuard  NeMo    LlamaGuard

cd benchmark/ui                      │

python3 app.py                      ▼

```                  Trylon ──► Ollama (Mistral)

```

Open: **http://localhost:5000**

**Everything runs in Docker with GPU support.**

## Tested Firewalls

---

1. **Rebuff** - Vector similarity detection

2. **PromptGuard** - Meta's instruction classifier## 📋 Prerequisites

3. **NeMo Guardrails** - NVIDIA's LLM-based rules

4. **Trylon Gateway** - PII detection API### Required

5. **LlamaGuard** - Meta's safety classifier- **Docker** + **Docker Compose** v2.0+

- **NVIDIA GPU** (tested on RTX 5090, works on any CUDA-capable GPU)

## Results Location- **NVIDIA Container Toolkit** installed

- **16GB+ RAM** (for Llama Guard 8B)

```bash- **50GB+ disk space** (for models)

results/

└── benchmark_run/### Verify GPU Access

    └── run_*/```bash

        ├── *_metrics.txt        # Accuracy, ASR, F1, etc.docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

        ├── *_summary.csv        # Per-prompt results```

        └── firewall_comparison.txt  # Full comparison

```You should see your GPU listed.



## Stop Services---



```bash## 🚀 Quick Start (5 Steps)

docker-compose down

```### Step 1️⃣: Clone and Setup


```bash
git clone <your-repo>
cd llm-firewall-benchmark

# Copy environment template
cp .env.example .env

# Edit .env if you want to add API keys (OPTIONAL)
nano .env
```

**Note:** Most firewalls work WITHOUT API keys! Only add keys if:
- You want NeMo to use GPT models (instead of local Ollama)
- You need Hugging Face gated models (for Llama Guard)

### Step 2️⃣: Start Docker Services

```bash
# Build and start all containers
docker-compose up -d

# Check status
docker-compose ps
```

Expected output:
```
NAME       IMAGE                              STATUS
ollama     ollama/ollama:latest              Up (healthy)
trylon     trustibleai/trylon-gateway        Up
benchmark  llm-firewall-benchmark            Up
```

### Step 3️⃣: Download Ollama Model (ONE TIME)

```bash
# Download Mistral model (4.1GB)
docker exec -it ollama ollama pull mistral

# Verify
docker exec -it ollama ollama list
```

### Step 4️⃣: Run the Benchmark

```bash
# Enter benchmark container
docker exec -it benchmark bash

# Run all firewalls
python3 main.py

# Or run specific firewalls only
python3 main.py --firewalls rebuff promptguard llamaguard
```

### Step 5️⃣: View Results

Results are saved to `./results/run_YYYYMMDD_HHMMSS/`

```bash
# List all runs
ls -lh results/

# View latest comparison
cat results/run_*/firewall_comparison.txt

# Check individual metrics
cat results/run_*/LlamaGuard_metrics.txt
```

---

## 📊 Understanding Results

### Metrics Explained

| Metric | What It Means | Good Value |
|--------|---------------|------------|
| **Accuracy** | Overall correctness | High (>0.90) |
| **Precision** | When it blocks, is it right? | High = fewer false alarms |
| **Recall** | % of attacks caught | High = better security |
| **F1 Score** | Balance of precision/recall | High (>0.85) |
| **Attack Success Rate** | % of attacks that got through | **LOW is better** (<10%) |
| **Latency** | Speed (ms per prompt) | Low = faster |

### Example Output

```
╔═══════════════╦══════════╦═══════════╦════════╦═════╦═══════════╦══════════════════╗
║ Firewall      ║ Accuracy ║ Precision ║ Recall ║ F1  ║ Latency   ║ Attack Success % ║
╠═══════════════╬══════════╬═══════════╬════════╬═════╬═══════════╬══════════════════╣
║ Rebuff        ║ 0.8800   ║ 0.9000    ║ 0.8500 ║ ... ║ 45.32 ms  ║ 15.00%           ║
║ PromptGuard   ║ 0.9200   ║ 0.9100    ║ 0.9300 ║ ... ║ 123.45 ms ║ 7.00%            ║
║ LlamaGuard    ║ 0.9600   ║ 0.9500    ║ 0.9700 ║ ... ║ 456.78 ms ║ 3.00%            ║
╚═══════════════╩══════════╩═══════════╩════════╩═════╩═══════════╩══════════════════╝
```

**LlamaGuard wins** (highest accuracy, lowest attack success rate) but is slower.

---

## 🔧 Advanced Usage

### Run Specific Firewalls

```bash
# Only fast firewalls
python3 main.py --firewalls rebuff promptguard

# Only AI-powered ones
python3 main.py --firewalls promptguard llamaguard

# Only Trylon + NeMo
python3 main.py --firewalls trylon nemo
```

### Custom Datasets

Add your own datasets to `benchmark/datasets/`:

```json
[
  {
    "id": 1001,
    "prompt": "Your custom attack prompt here",
    "label": "attack",
    "category": "custom"
  }
]
```

Then reload and rerun.

### Export Results

```bash
# Copy results to host
docker cp benchmark:/app/results ./exported_results

# Or mount different directory
docker-compose down
# Edit docker-compose.yml volumes
docker-compose up -d
```

---

## 🐛 Troubleshooting

### Issue: "CUDA out of memory"

**Solution:** Reduce batch size or run firewalls separately

```bash
# Run one at a time
python3 main.py --firewalls rebuff
python3 main.py --firewalls promptguard
python3 main.py --firewalls llamaguard
```

### Issue: "Ollama connection failed"

**Solution:** Check Ollama is running and model is downloaded

```bash
docker logs ollama
docker exec -it ollama ollama list
docker exec -it ollama ollama pull mistral
```

### Issue: "Trylon health check failing"

**Solution:** Some Trylon versions don't have `/health` endpoint

```bash
# Check if Trylon is actually running
docker logs trylon

# Test manually
curl http://localhost:8080/
```

This is expected and won't block the benchmark.

### Issue: "PromptGuard model download fails"

**Solution:** Need Hugging Face token for gated models

1. Get token from https://huggingface.co/settings/tokens
2. Add to `.env`: `HF_TOKEN=hf_xxxxx`
3. Rebuild: `docker-compose up -d --build`

### Issue: "NeMo Guardrails errors"

**Solution:** NeMo config is complex. Check:

```bash
# Verify config exists
ls -la benchmark/config/nemo/

# Simplify config or skip NeMo
python3 main.py --firewalls rebuff promptguard llamaguard
```

---

## 📁 Project Structure

```
llm-firewall-benchmark/
├── docker-compose.yml          # Docker orchestration
├── .env                        # API keys (optional)
├── .env.example                # Template
├── README.md                   # This file
│
├── benchmark/
│   ├── Dockerfile              # GPU-enabled Python container
│   ├── requirements.txt        # Python dependencies
│   ├── main.py                 # Main benchmark runner
│   │
│   ├── datasets/               # Curated attack datasets
│   │   ├── prompt_injection.json
│   │   ├── jailbreak.json
│   │   └── safe_prompts.json
│   │
│   ├── firewalls/              # Firewall adapters
│   │   ├── base.py
│   │   ├── rebuff_fw.py
│   │   ├── promptguard_fw.py
│   │   ├── nemo_fw.py
│   │   ├── trylon_fw.py
│   │   └── llamaguard_fw.py
│   │
│   ├── evaluation/             # Metrics & logging
│   │   ├── metrics.py
│   │   └── logger.py
│   │
│   └── config/
│       └── nemo/
│           └── config.yml
│
└── results/                    # Auto-generated results
    └── run_YYYYMMDD_HHMMSS/
        ├── summary.json
        ├── firewall_comparison.txt
        ├── *_metrics.txt
        └── *_results.jsonl
```

---

## 🔬 How It Works

### 1. Dataset Loading
Loads 50 curated prompts (30 attacks + 20 safe) from JSON files.

### 2. Firewall Initialization
Each firewall adapter:
- Loads models (PromptGuard, Llama Guard)
- Connects to services (Trylon, Ollama)
- Sets up rules (NeMo, Rebuff)

### 3. Evaluation Loop
For each prompt:
1. Send to firewall
2. Get decision (ALLOW/BLOCK)
3. Measure latency
4. Log result

### 4. Metrics Calculation
Uses sklearn to compute:
- Confusion matrix (TP, TN, FP, FN)
- Classification metrics (accuracy, precision, recall, F1)
- Performance metrics (latency, error rate)

### 5. Report Generation
Generates:
- Individual firewall reports
- Comparison table
- JSON summary
- CSV exports

---

## 🎓 Research Use

### Citation

If you use this benchmark in research:

```bibtex
@software{llm_firewall_benchmark,
  title={LLM Firewall Benchmark Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/llm-firewall-benchmark}
}
```

### Reproducibility

- All datasets are manually curated (no scraping)
- Ground truth labels included
- Docker ensures consistent environment
- Seeds fixed for deterministic results

---

## 🛠️ Development

### Adding a New Firewall

1. Create `benchmark/firewalls/mynew_fw.py`
2. Inherit from `Firewall` base class
3. Implement `initialize()` and `_evaluate_impl()`
4. Add to `main.py` available_firewalls dict
5. Run: `python3 main.py --firewalls mynew`

### Adding New Datasets

1. Create JSON file in `benchmark/datasets/`
2. Format: `[{"id": 1, "prompt": "...", "label": "attack|safe"}]`
3. Rerun benchmark

### Changing Metrics

Edit `benchmark/evaluation/metrics.py` to add custom metrics.

---

## 📜 License

MIT License - See LICENSE file

---

## 🙏 Acknowledgments

- **Meta** - PromptGuard & Llama Guard models
- **NVIDIA** - NeMo Guardrails & CUDA support
- **Rebuff** - Prompt injection detection
- **Trylon** - Gateway architecture
- **Ollama** - Local LLM serving

---

## 🚨 Ethical Use

This tool is for **defensive security research** only:

✅ Testing firewall effectiveness  
✅ Improving LLM safety  
✅ Academic research  

❌ Developing attacks  
❌ Bypassing safety measures  
❌ Malicious use  

**Use responsibly.**

---

## 📞 Support

Issues? Questions?

1. Check **Troubleshooting** section above
2. Review logs: `docker logs benchmark`
3. Open a GitHub issue

---

**Built with ❤️ for LLM Security Research**
