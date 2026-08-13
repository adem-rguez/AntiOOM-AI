import requests
import json

BASE_URL = "http://localhost:8080"

def test_aiatm_daemon():
    print("==================================================")
    print("⚡ TESTING AIATM LOCAL INFERENCE DAEMON ⚡")
    print("==================================================")

    # 1. Health Check
    try:
        res = requests.get(f"{BASE_URL}/health")
        print(f"\n[1] Health Check Response ({res.status_code}): {res.text}")
    except Exception as e:
        print(f"\n❌ Could not connect to daemon at {BASE_URL}: {e}")
        print("💡 Make sure to compile and run the daemon first using: cargo run --bin daemon-core")
        return

    # 2. Hardware System Info
    res = requests.get(f"{BASE_URL}/v1/system/info")
    print(f"\n[2] System Info:")
    print(json.dumps(res.json(), indent=2))

    # 3. Model Fit Estimator for Qwen3.5-0.8B-Q8_0.gguf
    fit_payload = {
        "parameter_count_billions": 0.8,
        "quantization": "Q8_0",
        "context_size": 4096,
        "modality": "Text"
    }
    res = requests.post(f"{BASE_URL}/v1/fit-estimator", json=fit_payload)
    print(f"\n[3] Pre-Download Model Fit Estimation for Qwen3.5-0.8B (Q8_0):")
    print(json.dumps(res.json(), indent=2))

    # 4. OpenAI Chat Completion Endpoint
    chat_payload = {
        "model": "models/Qwen3.5-0.8B-Q8_0.gguf",
        "messages": [
            {"role": "user", "content": "Explain quantum computing in one sentence."}
        ],
        "temperature": 0.7,
        "max_tokens": 128
    }
    res = requests.post(f"{BASE_URL}/v1/chat/completions", json=chat_payload)
    print(f"\n[4] Chat Completion Result:")
    print(json.dumps(res.json(), indent=2))

    # 5. Image Generation Endpoint
    img_payload = {
        "prompt": "Cyberpunk city illuminated with neon lights"
    }
    res = requests.post(f"{BASE_URL}/v1/images/generations", json=img_payload)
    print(f"\n[5] Image Generation Result:")
    print(f"Created: {res.json().get('created')}, Image Base64 length: {len(res.json()['data'][0]['b64_json'])} chars")

    # 6. MoE Stats
    res = requests.get(f"{BASE_URL}/v1/moe/stats")
    print(f"\n[6] MoE Cache Stats:")
    print(json.dumps(res.json(), indent=2))

    # 7. Cluster LAN Nodes
    res = requests.get(f"{BASE_URL}/v1/cluster/nodes")
    print(f"\n[7] LAN Peer Nodes:")
    print(json.dumps(res.json(), indent=2))

    print("\n==================================================")
    print("✅ TEST SUITE COMPLETED")
    print("==================================================")

if __name__ == "__main__":
    test_aiatm_daemon()
