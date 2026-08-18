"""
HF Transformers OpenAI-compatible HTTP server for safetensors text LLMs.

Spawned as a subprocess by the Rust daemon-core `load_model` handler. Loads a
`transformers` `AutoModelForCausalLM` once at startup and serves chat
completions over a stdlib HTTP server, speaking the same `/v1/chat/completions`
contract as `llama-server.exe` so the existing chat proxy routes to it
unchanged.

Usage:
    python hf_transformers_server.py --model-path <path/to/model_dir> --port <port>

`--model-path` must be a DIRECTORY containing config.json, *.safetensors and
tokenizer files (i.e. what `from_pretrained` expects).

Prints "READY" to stdout once the model is loaded and the server is
listening. The Rust side watches for this line.
"""

import argparse
import json
import os
import sys
import threading
import time
import uuid

import numpy as np  # noqa: F401

# Import torch up front: creating a CUDA session with a different DLL-loading
# order than torch expects can fail with WinError 127. Loading torch first
# and injecting the nvidia package DLL directories wins. Mirrors
# kokoro_tts_server.py's import ordering.
try:
    import torch  # noqa: F401
except ImportError:
    pass

if sys.platform == "win32":
    import importlib.util as _ilu
    for _mod in ("nvidia.cu13", "nvidia.cudnn"):
        _spec = _ilu.find_spec(_mod)
        if _spec and _spec.submodule_search_locations:
            for _loc in _spec.submodule_search_locations:
                for _bin in ("bin", os.path.join("bin", "x86_64")):
                    _p = os.path.join(_loc, _bin)
                    if os.path.isdir(_p):
                        os.environ["PATH"] = _p + ";" + os.environ.get("PATH", "")
                        os.add_dll_directory(_p)

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
except ImportError as e:
    print(f"Failed to import transformers/torch: {e}", file=sys.stderr)
    sys.exit(1)

MODEL = None
TOKENIZER = None


def build_prompt(messages: list) -> str:
    if TOKENIZER.chat_template:
        return TOKENIZER.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    parts = []
    for m in messages:
        parts.append(f"{m.get('role', 'user')}: {m.get('content', '')}")
    parts.append("assistant:")
    return "\n".join(parts)


def generation_kwargs(req: dict) -> dict:
    max_new_tokens = int(req.get("max_tokens") or 512)
    temperature = req.get("temperature")
    top_p = req.get("top_p")

    kwargs = {"max_new_tokens": max_new_tokens}
    if temperature is None or float(temperature) == 0:
        kwargs["do_sample"] = False
    else:
        kwargs["do_sample"] = True
        kwargs["temperature"] = float(temperature)
        if top_p is not None:
            kwargs["top_p"] = float(top_p)
    return kwargs


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

    def do_GET(self):
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self._send_json(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
            messages = req.get("messages", [])
            if not messages:
                self._send_json(400, {"error": "'messages' is required"})
                return

            prompt = build_prompt(messages)
            inputs = TOKENIZER(prompt, return_tensors="pt").to(MODEL.device)
            input_len = inputs["input_ids"].shape[1]

            if req.get("stream"):
                self._handle_stream(inputs, input_len, req)
            else:
                self._handle_non_stream(inputs, input_len, req)
        except Exception as e:
            try:
                self._send_json(500, {"error": str(e)})
            except Exception:
                pass

    def _handle_non_stream(self, inputs, input_len, req):
        with torch.no_grad():
            output_ids = MODEL.generate(**inputs, **generation_kwargs(req))
        new_tokens = output_ids[0][input_len:]
        text = TOKENIZER.decode(new_tokens, skip_special_tokens=True)

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        self._send_json(200, {
            "id": completion_id,
            "object": "chat.completion",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(new_tokens),
                "total_tokens": input_len + len(new_tokens),
            },
        })

    def _handle_stream(self, inputs, input_len, req):
        streamer = TextIteratorStreamer(
            TOKENIZER, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = generation_kwargs(req)
        gen_kwargs.update(inputs)
        gen_kwargs["streamer"] = streamer

        thread = threading.Thread(target=self._generate_in_thread, args=(gen_kwargs,))
        thread.start()

        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        try:
            for piece in streamer:
                if not piece:
                    continue
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }],
                }
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode("utf-8"))
                self.wfile.flush()

            final_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            self.wfile.write(f"data: {json.dumps(final_chunk)}\n\n".encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            thread.join()

    def _generate_in_thread(self, gen_kwargs):
        with torch.no_grad():
            MODEL.generate(**gen_kwargs)


def main():
    global MODEL, TOKENIZER

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    print(f"Loading transformers model from {args.model_path}...", file=sys.stderr)

    TOKENIZER = AutoTokenizer.from_pretrained(args.model_path)
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype="auto", device_map="cuda"
    )
    MODEL.eval()

    # Refuse to share a port with an orphaned server from an earlier daemon run —
    # on Windows SO_REUSEADDR would let the bind succeed and split requests between
    # the two processes, so requests silently land on the stale one.
    class ExclusiveHTTPServer(ThreadingHTTPServer):
        allow_reuse_address = False

    server = ExclusiveHTTPServer(("127.0.0.1", args.port), Handler)
    print("READY", flush=True)
    sys.stdout.flush()

    server.serve_forever()


if __name__ == "__main__":
    main()
