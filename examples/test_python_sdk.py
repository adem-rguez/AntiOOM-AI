import sys
import os

# Add sdk/python to path for testing (dispos_sdk module)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sdk", "python")))

from dispos_sdk import DisposClient

def main():
    client = DisposClient("http://localhost:8080")
    
    print("Testing DisposAI Python SDK...")
    health = client.health()
    print(f"Health: {health}")
    
    fit = client.estimate_fit(
        parameter_count_billions=0.8,
        quantization="Q8_0",
        context_size=4096
    )
    print(f"Fit Estimation: Fits in VRAM = {fit['fits_in_vram']}, Estimated Speed = {fit['estimated_tok_per_sec']} tok/s")
    
    chat = client.chat_completion(
        model="models/Qwen3.5-0.8B-Q8_0.gguf",
        messages=[{"role": "user", "content": "Hello DisposAI!"}]
    )
    print(f"Completion Response: {chat['choices'][0]['message']['content']}")

if __name__ == "__main__":
    main()
