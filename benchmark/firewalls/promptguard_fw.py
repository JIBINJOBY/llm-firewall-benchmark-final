"""
PromptGuard Firewall Adapter
Uses Meta's PromptGuard model from Hugging Face for jailbreak detection
Model: meta-llama/Prompt-Guard-86M
"""

from firewalls.base import Firewall, FirewallResult
import torch
import os


class PromptGuardFirewall(Firewall):
    """
    PromptGuard: Meta's model for detecting prompt injection and jailbreak attempts
    Uses Hugging Face transformers to load the model directly
    Note: Requires HF_TOKEN for gated model access
    """

    def __init__(self, threshold: float = 0.5, model_name: str = "meta-llama/Prompt-Guard-86M"):
        super().__init__(name="PromptGuard")
        self.threshold = threshold
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        # PromptGuard runs on CPU (86M params, very fast)
        # RTX 5090 (sm_120) not yet supported by PyTorch as of 2.6.0
        self.device = "cpu"

    def initialize(self) -> bool:
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Check for HF token (required for gated model)
            hf_token = os.getenv("HF_TOKEN")
            if not hf_token:
                print(f"⚠️  {self.name}: HF_TOKEN not set for gated model access")
                print(f"   Steps to get access:")
                print(f"   1. Go to https://huggingface.co/{self.model_name}")
                print(f"   2. Request access (instant approval)")
                print(f"   3. Get token from https://huggingface.co/settings/tokens")
                print(f"   4. Add to .env file: HF_TOKEN=your_token_here")
                print(f"   5. Or run: docker exec benchmark huggingface-cli login")
                return False
            
            print(f"🔄 Loading {self.name} model from Hugging Face ({self.model_name})...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, token=hf_token)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, token=hf_token)
            self.model.to(self.device)
            self.model.eval()
            
            self._initialized = True
            print(f"✅ {self.name} initialized on {self.device} (threshold={self.threshold})")
            return True
        except Exception as e:
            print(f"❌ {self.name} initialization failed: {e}")
            return False

    def _evaluate_impl(self, prompt: str) -> FirewallResult:
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # Get prediction (assuming binary classification: 0=safe, 1=attack)
            attack_prob = probs[0][1].item()
            is_attack = attack_prob >= self.threshold
            
            decision = "BLOCK" if is_attack else "ALLOW"
            
            return FirewallResult(
                decision=decision,
                confidence=attack_prob,
                latency_ms=0.0,
                firewall_name=self.name,
                metadata={
                    "attack_probability": attack_prob,
                    "safe_probability": probs[0][0].item(),
                    "threshold": self.threshold,
                    "device": self.device
                }
            )
        except Exception as e:
            return FirewallResult(
                decision="ERROR",
                confidence=0.0,
                latency_ms=0.0,
                firewall_name=self.name,
                metadata={"error": str(e)}
            )
    
    def cleanup(self):
        """Free GPU memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
