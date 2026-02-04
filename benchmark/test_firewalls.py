#!/usr/bin/env python3
"""
Quick firewall test script
Tests each firewall individually with a simple prompt
"""

import sys

def test_rebuff():
    print("\n" + "="*80)
    print("Testing Rebuff (requires API key)")
    print("="*80)
    try:
        from firewalls.rebuff_fw import RebuffFirewall
        fw = RebuffFirewall()
        if fw.initialize():
            result = fw.evaluate("Hello world")
            print(f"✅ Rebuff works: {result.decision} (confidence: {result.confidence})")
            return True
        else:
            print("❌ Rebuff initialization failed")
            return False
    except Exception as e:
        print(f"❌ Rebuff error: {e}")
        return False

def test_promptguard():
    print("\n" + "="*80)
    print("Testing PromptGuard (HuggingFace model)")
    print("="*80)
    try:
        from firewalls.promptguard_fw import PromptGuardFirewall
        fw = PromptGuardFirewall()
        if fw.initialize():
            result = fw.evaluate("Hello world")
            print(f"✅ PromptGuard works: {result.decision} (confidence: {result.confidence})")
            return True
        else:
            print("❌ PromptGuard initialization failed")
            return False
    except Exception as e:
        print(f"❌ PromptGuard error: {e}")
        return False

def test_nemo():
    print("\n" + "="*80)
    print("Testing NeMo Guardrails")
    print("="*80)
    try:
        from firewalls.nemo_fw import NeMoGuardrailsFirewall
        fw = NeMoGuardrailsFirewall()
        if fw.initialize():
            result = fw.evaluate("Hello world")
            print(f"✅ NeMo works: {result.decision} (confidence: {result.confidence})")
            return True
        else:
            print("❌ NeMo initialization failed")
            return False
    except Exception as e:
        print(f"❌ NeMo error: {e}")
        return False

def test_llamaguard():
    print("\n" + "="*80)
    print("Testing Llama Guard (Ollama)")
    print("="*80)
    try:
        from firewalls.llamaguard_fw import LlamaGuardFirewall
        fw = LlamaGuardFirewall()
        if fw.initialize():
            result = fw.evaluate("Hello world")
            print(f"✅ Llama Guard works: {result.decision} (confidence: {result.confidence})")
            return True
        else:
            print("❌ Llama Guard initialization failed")
            return False
    except Exception as e:
        print(f"❌ Llama Guard error: {e}")
        return False

def test_trylon():
    print("\n" + "="*80)
    print("Testing Trylon (Gateway)")
    print("="*80)
    try:
        from firewalls.trylon_fw import TrylonFirewall
        fw = TrylonFirewall()
        if fw.initialize():
            result = fw.evaluate("Hello world")
            print(f"✅ Trylon works: {result.decision} (confidence: {result.confidence})")
            return True
        else:
            print("❌ Trylon initialization failed")
            return False
    except Exception as e:
        print(f"❌ Trylon error: {e}")
        return False

if __name__ == "__main__":
    print("🔥 Firewall Integration Test")
    print("Testing each firewall individually...\n")
    
    results = {
        "Rebuff": test_rebuff(),
        "PromptGuard": test_promptguard(),
        "NeMo": test_nemo(),
        "Llama Guard": test_llamaguard(),
        "Trylon": test_trylon()
    }
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    for name, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{name:20s} {status}")
    
    working = sum(1 for v in results.values() if v)
    print(f"\n{working}/{len(results)} firewalls working")
