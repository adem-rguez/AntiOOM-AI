"""
ONNX TTS HTTP server.

Spawned as a subprocess by the Rust `tts-backend` crate. Loads an ONNX TTS
model once at startup and serves synthesis requests over a stdlib HTTP server.

Usage:
    python kokoro_tts_server.py --model-path <path/to/model.onnx> --port <port>

Protocol:
    POST /synthesize
        body: {"text": "...", "voice": "af_heart", "speed": 1.0}
        200:  {"audio_b64": "<base64 wav>", "sample_rate": 24000, "duration_s": 3.9}
        4xx/5xx: {"error": "..."}

Prints "READY" to stdout once the model is loaded and the server is
listening. The Rust backend watches for this line.
"""

import argparse
import base64
import io
import json
import os
import sys
import types
import wave

import numpy as np

os.environ["PATH"] = os.environ.get("PATH", "") + r";C:\Program Files\eSpeak NG"
sys.modules.setdefault("spacy", types.ModuleType("spacy"))

import onnxruntime as ort
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from huggingface_hub import hf_hub_download
from misaki.espeak import EspeakG2P

SESSION = None
G2P = None
VOCAB = None
VOICE_CACHE = {}


def load_voice(voice: str) -> np.ndarray:
    if voice in VOICE_CACHE:
        return VOICE_CACHE[voice]
    import torch
    voice_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename=f"voices/{voice}.pt")
    pack = torch.load(voice_path, weights_only=True).numpy()
    VOICE_CACHE[voice] = pack
    return pack


def synthesize(text: str, voice: str, speed: float) -> tuple[bytes, int, float]:
    phonemes = G2P(text)
    input_ids = [0] + [VOCAB[p] for p in phonemes if p in VOCAB] + [0]

    input_ids_np = np.array([input_ids], dtype=np.int64)
    pack = load_voice(voice)
    style = pack[len(phonemes) - 1]
    speed_np = np.array([speed], dtype=np.float32)

    result = SESSION.run(None, {
        "input_ids": input_ids_np,
        "style": style,
        "speed": speed_np,
    })
    waveform = result[0][0]

    sample_rate = 24000
    duration_s = len(waveform) / sample_rate
    pcm = (np.clip(waveform, -1.0, 1.0) * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return buf.getvalue(), sample_rate, duration_s


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/synthesize":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))

            text = req.get("text", "")
            voice = req.get("voice", "af_heart")
            speed = float(req.get("speed", 1.0))

            if not text:
                self._send_json(400, {"error": "'text' is required"})
                return

            wav_bytes, sample_rate, duration_s = synthesize(text, voice, speed)
            audio_b64 = base64.b64encode(wav_bytes).decode("ascii")

            self._send_json(200, {
                "audio_b64": audio_b64,
                "sample_rate": sample_rate,
                "duration_s": duration_s,
            })
        except Exception as e:
            self._send_json(500, {"error": str(e)})

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})


def main():
    global SESSION, G2P, VOCAB

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading ONNX model from {args.model_path}...", file=sys.stderr)

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    SESSION = ort.InferenceSession(args.model_path, providers=providers)
    active = SESSION.get_providers()
    print(f"ONNX providers: {active}", file=sys.stderr)

    config_path = hf_hub_download(repo_id="hexgrad/Kokoro-82M", filename="config.json")
    with open(config_path, encoding="utf-8") as f:
        VOCAB = json.load(f)["vocab"]

    G2P = EspeakG2P(language="en-us")

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print("READY", flush=True)
    sys.stdout.flush()

    server.serve_forever()


if __name__ == "__main__":
    main()
