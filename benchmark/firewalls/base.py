"""
Base Firewall Interface
All firewall adapters must implement this interface
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from dataclasses import dataclass
import time


@dataclass
class FirewallResult:
    """Standardized result format for all firewalls"""
    decision: str  # "ALLOW" or "BLOCK"
    confidence: float  # 0.0 to 1.0
    latency_ms: float  # Time taken in milliseconds
    firewall_name: str
    metadata: Dict[str, Any] = None  # Additional firewall-specific info
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class Firewall(ABC):
    """Abstract base class for all firewall implementations"""
    
    def __init__(self, name: str):
        self.name = name
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the firewall (load models, connect to APIs, etc.)
        Returns True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def _evaluate_impl(self, prompt: str) -> FirewallResult:
        """
        Internal evaluation logic - must be implemented by subclasses
        Should NOT handle timing - that's done by evaluate()
        """
        pass
    
    def evaluate(self, prompt: str) -> FirewallResult:
        """
        Public evaluation method with timing
        """
        if not self._initialized:
            raise RuntimeError(f"Firewall {self.name} not initialized. Call initialize() first.")
        
        start_time = time.time()
        
        try:
            result = self._evaluate_impl(prompt)
            result.latency_ms = (time.time() - start_time) * 1000
            return result
        except Exception as e:
            # Return error result on failure
            return FirewallResult(
                decision="ERROR",
                confidence=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                firewall_name=self.name,
                metadata={"error": str(e)}
            )
    
    def cleanup(self):
        """
        Optional cleanup method for resources
        """
        pass
    
    def __repr__(self):
        return f"<{self.__class__.__name__}: {self.name}>"
