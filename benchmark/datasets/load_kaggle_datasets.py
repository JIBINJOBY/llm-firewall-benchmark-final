"""
Kaggle Dataset Loader
Downloads and processes datasets from Kaggle
"""

import json
import os
import pandas as pd
import kagglehub
from pathlib import Path


def load_kaggle_prompt_injection(output_file: str = "kaggle_injection.json", max_samples: int = None):
    """
    Load prompt injection dataset from Kaggle (arielzilber/prompt-injection-in-the-wild)
    """
    print(f"📥 Loading Kaggle prompt injection dataset (arielzilber/prompt-injection-in-the-wild)...")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("arielzilber/prompt-injection-in-the-wild")
        print(f"📁 Dataset downloaded to: {path}")
        
        # Find and read the dataset files
        dataset_path = Path(path)
        injection_prompts = []
        
        # Look for CSV, JSON, or TXT files
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                print(f"   Found file: {file_path.name}")
                
                try:
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                        print(f"   CSV columns: {list(df.columns)}")
                        
                        # Try to find prompt column
                        prompt_col = None
                        for col in df.columns:
                            if 'prompt' in col.lower() or 'text' in col.lower() or 'injection' in col.lower():
                                prompt_col = col
                                break
                        
                        if prompt_col:
                            for idx, row in df.iterrows():
                                if max_samples and len(injection_prompts) >= max_samples:
                                    break
                                    
                                prompt = str(row[prompt_col]).strip()
                                if prompt and len(prompt) > 10 and prompt != 'nan':
                                    injection_prompts.append({
                                        "prompt": prompt,
                                        "label": "injection",
                                        "source": "kaggle/prompt-injection-in-the-wild",
                                        "metadata": {
                                            "file": file_path.name
                                        }
                                    })
                    
                    elif file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        if isinstance(data, list):
                            for item in data:
                                if max_samples and len(injection_prompts) >= max_samples:
                                    break
                                    
                                # Try different field names
                                prompt = item.get('prompt') or item.get('text') or item.get('injection')
                                if prompt and len(str(prompt).strip()) > 10:
                                    injection_prompts.append({
                                        "prompt": str(prompt).strip(),
                                        "label": "injection",
                                        "source": "kaggle/prompt-injection-in-the-wild",
                                        "metadata": {
                                            "file": file_path.name
                                        }
                                    })
                    
                    elif file_path.suffix == '.txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        for line in lines:
                            if max_samples and len(injection_prompts) >= max_samples:
                                break
                                
                            line = line.strip()
                            if line and len(line) > 10:
                                injection_prompts.append({
                                    "prompt": line,
                                    "label": "injection",
                                    "source": "kaggle/prompt-injection-in-the-wild",
                                    "metadata": {
                                        "file": file_path.name
                                    }
                                })
                
                except Exception as e:
                    print(f"   ⚠️  Could not parse {file_path.name}: {e}")
                    continue
        
        # Save to JSON in data subdirectory
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(injection_prompts, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Loaded {len(injection_prompts)} prompt injection samples from Kaggle")
        print(f"💾 Saved to: {output_path}")
        return injection_prompts
        
    except Exception as e:
        print(f"❌ Error loading Kaggle injection dataset: {e}")
        return []


def load_kaggle_jailbreak(output_file: str = "kaggle_jailbreak.json", max_samples: int = None):
    """
    Load jailbreak dataset from Kaggle (faiyazabdullah/jailbreaktracer-corpus)
    """
    print(f"📥 Loading Kaggle jailbreak dataset (faiyazabdullah/jailbreaktracer-corpus)...")
    
    try:
        # Download the dataset
        path = kagglehub.dataset_download("faiyazabdullah/jailbreaktracer-corpus")
        print(f"📁 Dataset downloaded to: {path}")
        
        # Find and read the dataset files
        dataset_path = Path(path)
        jailbreak_prompts = []
        
        # Look for CSV, JSON, or TXT files
        for file_path in dataset_path.rglob("*"):
            if file_path.is_file():
                print(f"   Found file: {file_path.name}")
                
                try:
                    if file_path.suffix == '.csv':
                        df = pd.read_csv(file_path)
                        print(f"   CSV columns: {list(df.columns)}")
                        
                        # Try to find prompt column
                        prompt_col = None
                        for col in df.columns:
                            if 'prompt' in col.lower() or 'text' in col.lower() or 'jailbreak' in col.lower():
                                prompt_col = col
                                break
                        
                        if prompt_col:
                            for idx, row in df.iterrows():
                                if max_samples and len(jailbreak_prompts) >= max_samples:
                                    break
                                    
                                prompt = str(row[prompt_col]).strip()
                                if prompt and len(prompt) > 20 and prompt != 'nan':
                                    jailbreak_prompts.append({
                                        "prompt": prompt[:2000],  # Limit length for complex jailbreaks
                                        "label": "jailbreak",
                                        "source": "kaggle/jailbreaktracer-corpus",
                                        "metadata": {
                                            "file": file_path.name
                                        }
                                    })
                    
                    elif file_path.suffix == '.json':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        if isinstance(data, list):
                            for item in data:
                                if max_samples and len(jailbreak_prompts) >= max_samples:
                                    break
                                    
                                # Try different field names
                                prompt = item.get('prompt') or item.get('text') or item.get('jailbreak')
                                if prompt and len(str(prompt).strip()) > 20:
                                    jailbreak_prompts.append({
                                        "prompt": str(prompt).strip()[:2000],
                                        "label": "jailbreak",
                                        "source": "kaggle/jailbreaktracer-corpus",
                                        "metadata": {
                                            "file": file_path.name
                                        }
                                    })
                    
                    elif file_path.suffix == '.txt':
                        with open(file_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            
                        for line in lines:
                            if max_samples and len(jailbreak_prompts) >= max_samples:
                                break
                                
                            line = line.strip()
                            if line and len(line) > 20:
                                jailbreak_prompts.append({
                                    "prompt": line[:2000],
                                    "label": "jailbreak",
                                    "source": "kaggle/jailbreaktracer-corpus",
                                    "metadata": {
                                        "file": file_path.name
                                    }
                                })
                
                except Exception as e:
                    print(f"   ⚠️  Could not parse {file_path.name}: {e}")
                    continue
        
        # Save to JSON in data subdirectory
        data_dir = Path(__file__).parent / "data"
        data_dir.mkdir(exist_ok=True)
        output_path = data_dir / output_file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(jailbreak_prompts, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Loaded {len(jailbreak_prompts)} jailbreak prompts from Kaggle")
        print(f"💾 Saved to: {output_path}")
        return jailbreak_prompts
        
    except Exception as e:
        print(f"❌ Error loading Kaggle jailbreak dataset: {e}")
        return []


def load_all_kaggle_datasets(injection_samples: int = None, jailbreak_samples: int = None):
    """
    Load both Kaggle datasets
    Use None for max samples to load all available prompts.
    """
    print("=" * 80)
    print("  LOADING DATASETS FROM KAGGLE")
    print("=" * 80)
    print()
    
    injection = load_kaggle_prompt_injection(max_samples=injection_samples)
    print()
    
    jailbreak = load_kaggle_jailbreak(max_samples=jailbreak_samples)
    print()
    
    print("=" * 80)
    print(f"📊 KAGGLE DATASETS SUMMARY:")
    print(f"   Prompt injections:       {len(injection)}")
    print(f"   Jailbreak prompts:       {len(jailbreak)}")
    print(f"   Total prompts:           {len(injection) + len(jailbreak)}")
    print("=" * 80)
    
    return {
        "injection": injection,
        "jailbreak": jailbreak
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load datasets from Kaggle")
    parser.add_argument("--injection", type=int, default=None, help="Number of injection samples (None=all)")
    parser.add_argument("--jailbreak", type=int, default=None, help="Number of jailbreak samples (None=all)")
    parser.add_argument("--all", action="store_true", help="Load ALL available prompts from each dataset")
    
    args = parser.parse_args()
    
    # If --all flag is used, set everything to None
    if args.all:
        injection_samples = None
        jailbreak_samples = None
    else:
        injection_samples = args.injection
        jailbreak_samples = args.jailbreak
    
    load_all_kaggle_datasets(
        injection_samples=injection_samples,
        jailbreak_samples=jailbreak_samples
    )
