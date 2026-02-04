"""
Combined Dataset Loader
Loads and combines datasets from HuggingFace and Kaggle
"""

import json
from pathlib import Path
from load_hf_datasets import load_all_datasets as load_hf
from load_kaggle_datasets import load_all_kaggle_datasets as load_kaggle


def load_combined_datasets(
    hf_jailbreak: int = None,
    hf_injection: int = None, 
    hf_benign: int = None,
    kaggle_jailbreak: int = None,
    kaggle_injection: int = None
):
    """
    Load and combine datasets from both HuggingFace and Kaggle.
    Use None for any parameter to load all available prompts from that source.
    
    Available datasets:
    - HuggingFace Jailbreak: 79 prompts
    - HuggingFace Injection: 203 prompts
    - HuggingFace Benign: 43,835 prompts
    - Kaggle Jailbreak: 93,904 prompts
    - Kaggle Injection: 86,572 prompts
    """
    
    print("=" * 80)
    print("  LOADING COMBINED DATASETS FROM HUGGINGFACE + KAGGLE")
    print("=" * 80)
    print()
    
    # Load HuggingFace datasets
    print("📦 Loading HuggingFace datasets...")
    hf_data = load_hf(
        jailbreak_samples=hf_jailbreak,
        injection_samples=hf_injection,
        benign_samples=hf_benign
    )
    print()
    
    # Load Kaggle datasets
    print("📦 Loading Kaggle datasets...")
    kaggle_data = load_kaggle(
        jailbreak_samples=kaggle_jailbreak,
        injection_samples=kaggle_injection
    )
    print()
    
    # Combine the datasets
    combined_jailbreak = hf_data["jailbreak"] + kaggle_data["jailbreak"]
    combined_injection = hf_data["injection"] + kaggle_data["injection"]
    combined_benign = hf_data["benign"]
    
    # Save combined datasets
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    
    with open(data_dir / "combined_jailbreak.json", 'w', encoding='utf-8') as f:
        json.dump(combined_jailbreak, f, indent=2, ensure_ascii=False)
    
    with open(data_dir / "combined_injection.json", 'w', encoding='utf-8') as f:
        json.dump(combined_injection, f, indent=2, ensure_ascii=False)
    
    with open(data_dir / "combined_benign.json", 'w', encoding='utf-8') as f:
        json.dump(combined_benign, f, indent=2, ensure_ascii=False)
    
    print("=" * 80)
    print("📊 COMBINED DATASET SUMMARY:")
    print(f"   Jailbreak prompts:       {len(combined_jailbreak):,}")
    print(f"   Prompt injections:       {len(combined_injection):,}")
    print(f"   Benign prompts:          {len(combined_benign):,}")
    print(f"   Total prompts:           {len(combined_jailbreak) + len(combined_injection) + len(combined_benign):,}")
    print("=" * 80)
    print()
    print("💾 Combined datasets saved to:")
    print(f"   - {data_dir}/combined_jailbreak.json")
    print(f"   - {data_dir}/combined_injection.json")
    print(f"   - {data_dir}/combined_benign.json")
    print("=" * 80)
    
    return {
        "jailbreak": combined_jailbreak,
        "injection": combined_injection,
        "benign": combined_benign
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Load and combine datasets from HuggingFace and Kaggle")
    
    # HuggingFace options
    parser.add_argument("--hf-jailbreak", type=int, default=None, help="HF jailbreak samples (None=all 79)")
    parser.add_argument("--hf-injection", type=int, default=None, help="HF injection samples (None=all 203)")
    parser.add_argument("--hf-benign", type=int, default=3000, help="HF benign samples (default=3000)")
    
    # Kaggle options
    parser.add_argument("--kaggle-jailbreak", type=int, default=None, help="Kaggle jailbreak samples (None=all 93,904)")
    parser.add_argument("--kaggle-injection", type=int, default=None, help="Kaggle injection samples (None=all 86,572)")
    
    # Quick options
    parser.add_argument("--all", action="store_true", help="Load ALL available prompts from all sources")
    parser.add_argument("--limit", type=int, default=None, help="Limit each category to N prompts")
    
    args = parser.parse_args()
    
    if args.all:
        # Load everything
        load_combined_datasets(
            hf_jailbreak=None,
            hf_injection=None,
            hf_benign=None,
            kaggle_jailbreak=None,
            kaggle_injection=None
        )
    elif args.limit:
        # Limit each category
        load_combined_datasets(
            hf_jailbreak=args.limit,
            hf_injection=args.limit,
            hf_benign=args.limit,
            kaggle_jailbreak=args.limit,
            kaggle_injection=args.limit
        )
    else:
        # Use provided arguments
        load_combined_datasets(
            hf_jailbreak=args.hf_jailbreak,
            hf_injection=args.hf_injection,
            hf_benign=args.hf_benign,
            kaggle_jailbreak=args.kaggle_jailbreak,
            kaggle_injection=args.kaggle_injection
        )
