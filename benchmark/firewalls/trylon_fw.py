"""
Trylon Firewall Adapter
Uses Trylon Gateway for LLM policy enforcement via HTTP API
"""

from firewalls.base import Firewall, FirewallResult
import httpx


class TrylonFirewall(Firewall):
    """
    Trylon: Gateway-based LLM firewall with policy enforcement
    Communicates via HTTP with the Trylon container
    """

    def __init__(self, base_url: str = "http://trylon:8000", timeout: float = 10.0):
        super().__init__(name="Trylon")
        self.base_url = base_url
        self.timeout = timeout
        self.client = None

    def initialize(self) -> bool:
        try:
            # Create HTTP client
            self.client = httpx.Client(timeout=self.timeout)
            
            # Test connection (optional - Trylon might not be running)
            health_url = f"{self.base_url}/health"
            try:
                response = self.client.get(health_url)
                if response.status_code == 200:
                    print(f"✅ {self.name} initialized (connected to {self.base_url})")
                    self._initialized = True
                    return True
            except Exception as e:
                # Connection failed - Trylon not available
                print(f"⚠️ {self.name} not available: {e}")
                print(f"   To enable Trylon:")
                print(f"   1. git clone https://github.com/trylonai/gateway.git trylon")
                print(f"   2. cd trylon && docker build -t trylon-gateway .")
                print(f"   3. Uncomment trylon service in docker-compose.yml")
                return False
                
        except Exception as e:
            print(f"❌ {self.name} initialization failed: {e}")
            return False

    def _evaluate_impl(self, prompt: str) -> FirewallResult:
        try:
            # Use Trylon's /safeguard endpoint for direct validation
            check_url = f"{self.base_url}/safeguard"
            
            payload = {
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            response = self.client.post(check_url, json=payload)
            
            # Trylon returns HTTP 200 for safe content and HTTP 400 for violations
            if response.status_code in [200, 400]:
                data = response.json()
                
                # Parse Trylon safeguard response
                # safety_code: 0 = SAFE, 10+ = violations detected
                # action: 0 = OVERRIDE (block), 1 = OBSERVE (log only), null for violations
                safety_code = data.get("safety_code", 0)
                action = data.get("action")
                message = data.get("message", "")
                
                # HTTP 400 means content was blocked (violation detected)
                if response.status_code == 400:
                    decision = "BLOCK"
                    confidence = min(safety_code / 100.0, 1.0) if safety_code > 0 else 0.8
                else:
                    # HTTP 200: check safety_code and action
                    is_violation = safety_code > 0
                    should_block = action == 0  # Action.OVERRIDE
                    decision = "BLOCK" if (is_violation and should_block) else "ALLOW"
                    confidence = min(safety_code / 100.0, 1.0) if is_violation else 0.0
                
                return FirewallResult(
                    decision=decision,
                    confidence=confidence,
                    latency_ms=0.0,
                    firewall_name=self.name,
                    metadata={
                        "safety_code": safety_code,
                        "action": action,
                        "message": message,
                        "http_status": response.status_code,
                        "raw_response": data
                    }
                )
            else:
                # Other HTTP errors (500, 503, etc.) - treat as error
                return FirewallResult(
                    decision="ERROR",
                    confidence=0.0,
                    latency_ms=0.0,
                    firewall_name=self.name,
                    metadata={
                        "error": f"HTTP {response.status_code}",
                        "response": response.text[:500]
                    }
                )
                
        except httpx.TimeoutException:
            return FirewallResult(
                decision="ERROR",
                confidence=0.0,
                latency_ms=0.0,
                firewall_name=self.name,
                metadata={"error": "Request timeout"}
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
        """Close HTTP client"""
        if self.client is not None:
            self.client.close()
