"""
NeMo Guardrails custom actions
Keyword-based detection without external LLM calls
"""

from nemoguardrails.actions import action


# Malicious keywords for detection
INJECTION_PATTERNS = [
    "ignore previous instructions",
    "ignore all previous",
    "disregard",
    "forget everything",
    "new instructions",
    "system prompt",
    "override",
    "admin mode",
    "developer mode",
    "god mode",
    "sudo",
    "jailbreak",
    "DAN mode",
    "roleplay as",
]


@action(is_system_action=True)
async def check_input_safety(context: dict):
    """
    Check if input contains known injection patterns
    Returns True if safe, False if malicious
    """
    user_message = context.get("user_message", "").lower()
    
    # Check for injection patterns
    for pattern in INJECTION_PATTERNS:
        if pattern in user_message:
            return False
    
    # Safe by default
    return True


@action(is_system_action=True)
async def check_output_safety(context: dict):
    """
    Check if output is safe
    Returns True if safe
    """
    # For now, allow all outputs
    return True
