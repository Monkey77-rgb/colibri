#!/usr/bin/env python3
"""Manual test fixture only (2026-09-13) -- NOT part of `make test`, not imported by any
unit test. Bridges the SERVE wire protocol (docs/serve_protocol.md) openai_server.py's
Engine class expects, to a REAL llama.cpp CPU backend, so the idle-sleep test protocol in
SLEEP_IDLE_2026-09-13.md can be run against a real loaded model and report real RSS / real
wake latency for a real .gguf, instead of a protocol-only mock that would report meaningless
numbers.

Why this exists at all: audited at b34cedb before writing it --
  - c/colibri.c (+ kimi_k3.c/inkling.c/deepseek_v4.c/olmoe.c), the only binaries that speak
    SERVE=1/SERVE_BATCH=1, require SNAP=<converted snapshot directory> in an
    architecture-specific tensor layout (colibri.c's is GLM/DeepSeek MLA:
    q_a_proj/kv_a_proj_with_mqa/...). None of them can load a standalone Qwen2.5 GGUF.
  - engine/coli and engine/banana (the Banana rewrite, engine/src/main.cpp) load a raw
    .gguf directly but implement NO serve protocol at all -- confirmed by `--help` and by
    grepping engine/src/ for SUBMIT/READY/SERVE (none found; docs/serve_protocol.md
    confirms the protocol lives only in glm.c).
  So there is currently no engine binary in this repo that both (a) speaks SERVE and
  (b) loads this Qwen2.5-3B GGUF. This bridge is a stand-in ENGINE for exactly that gap:
  it launches ~/opt/llama-cpp-native/bin/llama-server (CPU only -- no -ngl passed) as its
  own child, proxies SUBMIT/STOP/CANCEL to its /completion endpoint, and streams the reply
  back as DATA/DONE frames. Its own RSS is negligible (a Python HTTP client); the real
  model RSS is measured on the llama-server grandchild's PID (see the test script).

This file changes NO C/C++ and is not shipped -- it exists to give openai_server.py's
Engine class something SERVE-shaped to spawn/sleep/wake for the measurement run.
"""
import http.client
import json
import os
import signal
import subprocess
import sys
import threading
import time

LLAMA_SERVER = os.environ.get("BRIDGE_LLAMA_SERVER",
                              os.path.expanduser("~/opt/llama-cpp-native/bin/llama-server"))
MODEL = os.environ["SNAP"]  # openai_server.py's Engine always sets SNAP=<model>
PORT = int(os.environ.get("BRIDGE_LLAMA_PORT", "0")) or None


def log(msg):
    sys.stderr.write("[bridge] %s\n" % msg)
    sys.stderr.flush()


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    port = PORT or free_port()
    threads = os.environ.get("BRIDGE_THREADS", os.environ.get("OMP_NUM_THREADS", "4"))
    cmd = [LLAMA_SERVER, "-m", MODEL, "--host", "127.0.0.1", "--port", str(port),
           "-c", "2048", "-t", threads, "--no-slots", "--log-disable"]
    # No -ngl anywhere in this command: CPU only, deliberately (owner constraint --
    # the desktop 4070 is reserved for another measurement).
    log("starting llama-server: %s" % " ".join(cmd))
    child = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def handle_term(*_):
        log("terminating llama-server child pid=%d" % child.pid)
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:
            child.kill()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_term)
    signal.signal(signal.SIGINT, handle_term)

    # Wait for llama-server's own readiness (poll /health) before declaring READY upstream.
    deadline = time.time() + 120
    ready = False
    while time.time() < deadline:
        if child.poll() is not None:
            log("llama-server exited during startup, code=%s" % child.poll())
            sys.exit(1)
        try:
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=2)
            conn.request("GET", "/health")
            resp = conn.getresponse()
            if resp.status == 200:
                ready = True
                conn.close()
                break
            conn.close()
        except OSError:
            pass
        time.sleep(0.1)
    if not ready:
        log("llama-server never became ready")
        sys.exit(1)
    log("llama-server ready on port %d, pid=%d" % (port, child.pid))

    out = sys.stdout.buffer
    inp = sys.stdin.buffer
    out.write(b"\x01\x01READY\x01\x01\n")
    out.write(b"STAT 0 0.00 0.0 0.0\n")
    out.flush()

    write_lock = threading.Lock()

    def emit(line_bytes):
        with write_lock:
            out.write(line_bytes)
            out.flush()

    def serve_request(rid, prompt, max_tokens, temperature, top_p):
        body = json.dumps({
            "prompt": prompt, "n_predict": max_tokens,
            "temperature": temperature, "top_p": top_p, "stream": False,
        }).encode()
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=300)
        try:
            conn.request("POST", "/completion", body=body,
                        headers={"Content-Type": "application/json"})
            resp = conn.getresponse()
            payload = json.loads(resp.read())
        except Exception as exc:
            emit(("ERROR %s BAD_REQUEST %s\n" % (rid, str(exc).replace("\n", " "))).encode())
            return
        finally:
            conn.close()
        text = payload.get("content", "")
        n_predicted = payload.get("tokens_predicted", len(text.split()))
        n_prompt = payload.get("tokens_evaluated", 0)
        data = text.encode("utf-8")
        emit(("DATA %s %d\n" % (rid, len(data))).encode() + data + b"\n")
        emit(("DONE %s STAT %d 0.0 0.0 0.0 %d 0\n"
             % (rid, n_predicted, n_prompt)).encode())

    while True:
        line = inp.readline()
        if not line:
            break
        fields = line.decode("utf-8", "replace").strip().split()
        if not fields:
            continue
        if fields[0] == "SUBMIT" and len(fields) >= 6:
            rid, slot, blen, max_tokens = fields[1], fields[2], int(fields[3]), int(fields[4])
            temperature, top_p = float(fields[5]), float(fields[6]) if len(fields) > 6 else 1.0
            payload = inp.read(blen)
            inp.read(1)  # trailing \n
            prompt = payload.decode("utf-8", "replace")
            threading.Thread(target=serve_request,
                             args=(rid, prompt, max_tokens, temperature, top_p),
                             daemon=True).start()
        # STOP/CANCEL: not implemented in this bridge -- the manual test protocol
        # here never exercises client cancellation, only sleep/wake.

    handle_term()


if __name__ == "__main__":
    main()
