"""
HuggingFace Dataset Loader
Downloads and processes datasets from HuggingFace Hub
"""

import json
import os
from datasets import load_dataset
from pathlib import Path


def load_jailbreak_dataset(output_file: str = "jailbreak_hf.json", max_samples: int = None):
    """
    Load jailbreak attempts from rubend18/ChatGPT-Jailbreak-Prompts
    Dataset contains 79 jailbreak prompts. Use max_samples=None for all prompts.
    """
    print(f"📥 Loading jailbreak dataset from HuggingFace (rubend18/ChatGPT-Jailbreak-Prompts)...")
    
    try:
        ds = load_dataset("rubend18/ChatGPT-Jailbreak-Prompts", split="train")
        
        jailbreak_prompts = []
        total_available = len(ds)
        
        for i, item in enumerate(ds):
            if max_samples and len(jailbreak_prompts) >= max_samples:
                break
            
            # Extract the jailbreak prompt from 'Prompt' field
            prompt = item.get("Prompt", "")
            
            if prompt and isinstance(prompt, str) and len(prompt.strip()) > 20:
                jailbreak_prompts.append({
                    "prompt": prompt.strip()[:2000],  # Increased length limit for complex jailbreaks
                    "label": "jailbreak",
                    "source": "rubend18/ChatGPT-Jailbreak-Prompts",
                    "metadata": {
                        "name": str(item.get("Name", "unknown"))[:100],
                        "jailbreak_score": item.get("Jailbreak Score", 0)
                    }
                })
        
        # Save to JSON in data subdirectory
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jailbreak_prompts, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Loaded {len(jailbreak_prompts)} jailbreak prompts (out of {total_available} available)")
        print(f"💾 Saved to: {output_path}")
        return jailbreak_prompts
        
    except Exception as e:
        print(f"❌ Error loading jailbreak dataset: {e}")
        return []


def load_prompt_injection_dataset(output_file: str = "prompt_injection_hf.json", max_samples: int = None):
    """
    Load prompt injection attacks from deepset/prompt-injections
    Dataset contains 203 injection prompts (label=1). Use max_samples=None for all prompts.
    """
    print(f"📥 Loading prompt injection dataset from HuggingFace (deepset/prompt-injections)...")
    
    try:
        ds = load_dataset("deepset/prompt-injections", split="train")
        
        injection_prompts = []
        for i, item in enumerate(ds):
            if max_samples and len(injection_prompts) >= max_samples:
                break
            
            # Extract the injection prompt
            prompt = item.get("text", item.get("prompt", ""))
            label = item.get("label", 1)  # 1 = injection, 0 = benign
            
            # Only include actual injections (label=1)
            if prompt and label == 1 and len(prompt.strip()) > 10:
                injection_prompts.append({
                    "prompt": prompt.strip(),
                    "label": "injection",
                    "source": "deepset/prompt-injections",
                    "metadata": {
                        "original_label": label
                    }
                })
        
        # Save to JSON in data subdirectory
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(injection_prompts, f, indent=2, ensure_ascii=False)
        
        total_in_dataset = sum(1 for item in ds if item.get("label", 1) == 1)
        print(f"✅ Loaded {len(injection_prompts)} prompt injection samples (out of {total_in_dataset} available)")
        print(f"💾 Saved to: {output_path}")
        return injection_prompts
        
    except Exception as e:
        print(f"❌ Error loading prompt injection dataset: {e}")
        return []


def load_benign_dataset(output_file: str = "safe_prompts_hf.json", max_samples: int = None):
    """
    Load benign/safe prompts from Anthropic/hh-rlhf
    Dataset contains 43,835 prompts. Use max_samples=None for all prompts.
    """
    print(f"📥 Loading benign dataset from HuggingFace (Anthropic/hh-rlhf)...")
    
    try:
        # Load the helpful dataset (benign human-assistant conversations)
        ds = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="train")
        
        benign_prompts = []
        total_available = len(ds)
        
        for i, item in enumerate(ds):
            if max_samples and len(benign_prompts) >= max_samples:
                break
            
            # Extract human prompts from the conversation
            chosen = item.get("chosen", "")
            
            # Parse the conversation to extract the first human message
            if "\n\nHuman:" in chosen:
                parts = chosen.split("\n\nHuman:")
                if len(parts) > 1:
                    # Get the first human message
                    human_msg = parts[1].split("\n\nAssistant:")[0].strip()
                    
                    if human_msg and len(human_msg) > 20 and len(human_msg) < 500:
                        benign_prompts.append({
                            "prompt": human_msg,
                            "label": "safe",
                            "source": "Anthropic/hh-rlhf",
                            "metadata": {
                                "category": "helpful-base"
                            }
                        })
        
        # Save to JSON in data subdirectory
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(benign_prompts, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Loaded {len(benign_prompts)} benign prompts (out of {total_available} available)")
        print(f"💾 Saved to: {output_path}")
        return benign_prompts
        
    except Exception as e:
        print(f"❌ Error loading benign dataset: {e}")
        return []


def load_all_datasets(jailbreak_samples: int = None, injection_samples: int = None, benign_samples: int = None):
    """
    Load all three datasets from HuggingFace
    Use None for max samples to load all available prompts from each dataset.
    
    Current dataset sizes:
    - Jailbreak: 79 prompts (rubend18/ChatGPT-Jailbreak-Prompts)
    - Injection: 203 prompts (deepset/prompt-injections)
    - Benign: 43,835 prompts (Anthropic/hh-rlhf)
    """
    print("=" * 80)
    print("  LOADING DATASETS FROM HUGGINGFACE")
    print("=" * 80)
    print()
    
    jailbreak = load_jailbreak_dataset(max_samples=jailbreak_samples)
    print()
    
    injection = load_prompt_injection_dataset(max_samples=injection_samples)
    print()
    
    benign = load_benign_dataset(max_samples=benign_samples)
    print()
    
    print("=" * 80)
    print(f"📊 SUMMARY:")
    print(f"   Jailbreak prompts:       {len(jailbreak)}")
    print(f"   Prompt injections:       {len(injection)}")
    print(f"   Benign prompts:          {len(benign)}")
    print(f"   Total prompts:           {len(jailbreak) + len(injection) + len(benign)}")
    print("=" * 80)
    
    return {
        "jailbreak": jailbreak,
        "injection": injection,
        "benign": benign
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load datasets from HuggingFace")
    parser.add_argument("--jailbreak", type=int, default=None, help="Number of jailbreak samples (None=all)")
    parser.add_argument("--injection", type=int, default=None, help="Number of injection samples (None=all)")
    parser.add_argument("--benign", type=int, default=None, help="Number of benign samples (None=all)")
    parser.add_argument("--all", action="store_true", help="Load ALL available prompts from each dataset")
    
    args = parser.parse_args()
    
    # If --all flag is used, set everything to None
    if args.all:
        jailbreak_samples = None
        injection_samples = None
        benign_samples = None
    else:
        jailbreak_samples = args.jailbreak
        injection_samples = args.injection
        benign_samples = args.benign
    
    load_all_datasets(
        jailbreak_samples=jailbreak_samples,
        injection_samples=injection_samples,
        benign_samples=benign_samples
    )
