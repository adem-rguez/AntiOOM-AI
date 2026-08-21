import requests
from typing import Dict, Any, Optional

class DisposClient:
    """Python Client SDK for DisposAI Local Inference Daemon REST & Fit Estimator APIs."""
    
    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> str:
        res = requests.get(f"{self.base_url}/health")
        res.raise_for_status()
        return res.text

    def estimate_fit(
        self,
        parameter_count_billions: float,
        quantization: str = "Q4_K_M",
        context_size: int = 4096,
        modality: str = "Text"
    ) -> Dict[str, Any]:
        payload = {
            "parameter_count_billions": parameter_count_billions,
            "quantization": quantization,
            "context_size": context_size,
            "modality": modality
        }
        res = requests.post(f"{self.base_url}/v1/fit-estimator", json=payload)
        res.raise_for_status()
        return res.json()

    def chat_completion(
        self,
        model: str,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Dict[str, Any]:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        res = requests.post(f"{self.base_url}/v1/chat/completions", json=payload)
        res.raise_for_status()
        return res.json()
