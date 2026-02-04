# Trylon Gateway: The Open Source Firewall for LLMs

[![Build Status](https://img.shields.io/github/actions/workflow/status/trylonai/gateway/ci.yml?branch=main)](https://github.com/trylonai/gateway/actions)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](./CONTRIBUTING.md)

Building production-grade LLM applications means managing risks: prompt injections, toxic responses, data leaks, and inconsistent behavior. **Trylon Gateway is an open-source AI Gateway that provides the essential guardrails to manage these risks, ensuring your AI applications are secure, reliable, and responsible.**

It acts as a high-performance, self-hosted proxy that sits between your application and models from **OpenAI, Google Gemini, and Anthropic Claude**, giving you complete control over every interaction.

### How It Works

Trylon intercepts API calls to and from LLMs. It applies your custom policies locally to validate, sanitize, or block requests and responses in real-time, all within the security of your own infrastructure.


### Why Choose Trylon Gateway?

*   **Defend Against Attacks:** Implement a critical defense layer against common vulnerabilities like **prompt injections**, insecure outputs, and data exfiltration.
*   **Ensure Data Privacy & Compliance:** Automatically find and **redact PII** (Personally Identifiable Information) *before* it leaves your network, helping you build compliant applications.
*   **Enforce Brand Safety:** Use the powerful policy engine to implement **content filtering**, blocking toxicity, profanity, and other undesirable language to protect your users and your brand.
*   **Unify Your AI Stack:** Act as a single **LLM firewall** and proxy for multiple providers. Swap models without changing your application's core safety logic.
*   **High-Performance & Self-Hosted:** Built with FastAPI, Trylon Gateway is fast. Running in your own infrastructure means you have complete control and data privacy.

### Use Cases
*   **Public-facing Chatbots:** Prevent users from submitting sensitive data and ensure the LLM does not respond with toxic or off-brand content.
*   **Internal AI Tools:** Protect against the accidental leakage of internal secrets, API keys, or project codenames in prompts or responses.
*   **Regulated Industries (Healthcare, Finance):** Build a foundational layer for HIPAA or GDPR compliance by implementing strong PII redaction and logging policies.
*   **Content Generation Platforms:** Ensure that AI-generated content adheres to community guidelines and safety standards.

## Quick Start

The fastest way to deploy your own AI Gateway.

**Prerequisites:** Docker and Docker Compose.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/trylonai/gateway.git
    cd gateway
    ```

2.  **Prepare your environment:**
    ```bash
    # Copy the example environment file. No edits are needed to run the quick start.
    cp .env.example .env

    # An example policies.yaml and docker-compose.yml are already present.
    ```

3.  **Launch the Gateway:**
    ```bash
    docker-compose up -d
    ```
    Note: The first launch will take several minutes. The gateway needs to download the machine learning models (~1.5GB+). Subsequent launches will be much faster because the models are stored in a persistent Docker volume (trylon_hf_cache).

    You can monitor the download progress and see when the application is ready by watching the logs:

    ```bash
    docker-compose logs -f
    ```

4.  **Test a Guardrail in Action**

    The default `policies.yaml` comes with a PII guardrail enabled to **block** any request containing an email address. Let's test it.
    ```bash
    curl -s -X POST "http://localhost:8000/v1/chat/completions" \
      -H "Authorization: Bearer $OPENAI_API_KEY" \
      -H "Content-Type: application/json" \
      -d '{
        "model": "gpt-3.5-turbo",
        "messages": [{"role": "user", "content": "My email address is test@example.com"}]
      }' | jq
    ```
    You'll see a response with `finish_reason: "content_filter"`, confirming the block. Congratulations! You've just seen a guardrail in action without writing any code.

## API Usage & Examples

For detailed, runnable examples, please see the `/examples` directory.

### 1. OpenAI Proxy (`POST /v1/chat/completions`)

Configure your OpenAI client to point to the Trylon Gateway.
```python
import os
from openai import OpenAI

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"), # Your REAL OpenAI key
    base_url="http://localhost:8000/v1"      # Your Trylon Gateway URL + /v1
)
```
> **Pro-Tip:** To reliably detect blocks, use `client.with_raw_response` to inspect the `X-Trylon-Blocked: true` header or check for `finish_reason: "content_filter"`.

### 2. Gemini & Claude Proxies

Trylon offers the same drop-in proxy capability for Google Gemini and Anthropic Claude.
<details>
<summary><b>Click for Gemini and Claude configuration examples</b></summary>

#### Gemini Proxy (`POST /v1beta/models/{model}:generateContent`)
```python
import google.generativeai as genai
genai.configure(
    api_key=os.environ.get("GEMINI_API_KEY"),
    client_options={"api_endpoint": "http://localhost:8000"},
    transport='rest'
)
```

#### Claude Proxy (`POST /anthropic/v1/messages`)
```python
from anthropic import Anthropic
client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY"),
    base_url="http://localhost:8000/anthropic"
)
```
</details>

### 3. Direct Validation (`POST /safeguard`)

For custom workflows, use the `/safeguard` endpoint to validate content without proxying.
```bash
curl -s -X POST "http://localhost:8000/safeguard" \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "My email is example@gmail.com."}
    ]
  }' | jq
```
**Response:**
```bash
{
  "safety_code": 10,
  "message": "Potentially sensitive information detected.",
  "action": 0
}
```

## Policy Configuration (`policies.yaml`)

This file is the heart of your AI Gateway. All rules are defined here.
**See the full [Policy Configuration Guide](./docs/POLICIES.md) for detailed mappings and advanced examples.**

## Contributing

We are building a community dedicated to making AI safer and more reliable. Contributions are welcome!
*   Check out the [open issues](https://github.com/trylonai/gateway/issues) to see where you can help.
*   Read our **[CONTRIBUTING.md](./CONTRIBUTING.md)** guide to get started.
*   Have an idea for a new guardrail? Open a feature request!

## Relationship to Trylon AI Cloud

This open-source project is the core data plane that powers the [Trylon AI](https://trylon.ai) commercial platform. Our cloud offering builds on this core to provide an enterprise-ready solution with: a UI for policy management, centralized observability, team collaboration (RBAC/SSO), and managed API key security.