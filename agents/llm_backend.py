"""
Local vision-language model backend (llama.cpp `llama-server`, Vulkan build) used by the unattended pipeline.
No API key, no cost. The server is started automatically when it is not running.

  ask_json(prompt, images=[jpg bytes], check=fn, retries=3) -> (object, error)

Files (downloaded once into A:\\llm):
  A:\\llm\\llama.cpp\\llama-server.exe
  A:\\llm\\models\\Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf  +  mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf

Environment overrides: LLM_URL (default http://127.0.0.1:8081), LLM_DIR (default A:\\llm), LLM_CTX (default 12288),
LLM_GPU_LAYERS (default 99; set 0 for CPU only).
"""
import base64
import json
import os
import re
import subprocess
import time

import requests

URL = os.environ.get("LLM_URL", "http://127.0.0.1:8081")
DIR = os.environ.get("LLM_DIR", r"A:\llm")
CTX = int(os.environ.get("LLM_CTX", "12288"))
NGL = os.environ.get("LLM_GPU_LAYERS", "99")
MODEL = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ = "mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf"

_proc = None


def _up():
    try:
        return requests.get(URL + "/health", timeout=3).json().get("status") == "ok"
    except Exception:
        return False


def ensure_server(timeout=240):
    """Start llama-server if it is not answering. Returns True when it is ready."""
    global _proc
    if _up():
        return True
    exe = os.path.join(DIR, "llama.cpp", "llama-server.exe")
    model, mmproj = os.path.join(DIR, "models", MODEL), os.path.join(DIR, "models", MMPROJ)
    if not (os.path.exists(exe) and os.path.exists(model) and os.path.exists(mmproj)):
        return False
    port = URL.rsplit(":", 1)[1]
    log = open(os.path.join(DIR, "llama-server.log"), "ab")
    cmd = [exe, "-m", model, "--mmproj", mmproj, "-c", str(CTX), "-ngl", NGL, "--port", port,
           "--host", "127.0.0.1", "-np", "1", "--no-webui"]
    img_min = os.environ.get("LLM_IMAGE_MIN_TOKENS", "").strip()
    if img_min:
        cmd += ["--image-min-tokens", img_min]
    fa = os.environ.get("LLM_FLASH_ATTN", "").strip()
    if fa:
        cmd += ["-fa", fa]
    _proc = subprocess.Popen(cmd, stdout=log, stderr=log,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    t0 = time.time()
    while time.time() - t0 < timeout:
        if _up():
            return True
        if _proc.poll() is not None:
            return False
        time.sleep(2)
    return False


def stop_server():
    """Stop the model server (also one started by an earlier process) to free RAM and VRAM."""
    global _proc
    if _proc is not None and _proc.poll() is None:
        _proc.terminate()
    _proc = None
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)


def _build_content(prompt, images=None):
    """Put images before text so llama-server can reuse the image KV prefix across sequential questions."""
    content = [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(im).decode("ascii")}}
               for im in (images or [])]
    content.append({"type": "text", "text": prompt})
    return content


def chat(prompt, images=None, json_mode=True, max_tokens=1500, temperature=0.1, timeout=1800):
    """One chat call. images: list of JPEG bytes. Returns the reply text."""
    if not ensure_server():
        raise RuntimeError("the local model server could not be started (is A:\\llm complete?)")
    content = _build_content(prompt, images)
    body = {"messages": [{"role": "user", "content": content}], "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    r = requests.post(URL + "/v1/chat/completions", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError("model server error %s: %s" % (r.status_code, r.text[:300]))
    return r.json()["choices"][0]["message"]["content"]


def yes_probability(prompt, images=None, timeout=300):
    """Probability that the model answers YES to a yes/no question (read from the log-probabilities of the first token).
    A continuous score is far more useful than the model's hard answer, which is biased."""
    import math
    if not ensure_server():
        raise RuntimeError("the local model server could not be started")
    content = _build_content(prompt, images)
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": 1,
            "logprobs": True, "top_logprobs": 10}
    r = requests.post(URL + "/v1/chat/completions", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError("model server error %s: %s" % (r.status_code, r.text[:200]))
    top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    py = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("y"))
    pn = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower().startswith("n"))
    return py / (py + pn) if (py + pn) > 0 else 0.0


def choice_probabilities(prompt, images, letters, timeout=300):
    """Multiple-choice question: the prompt lists the options as 'A) ...', 'B) ...'. Returns {letter: probability} from the
    log-probabilities of the first answer token, which is far more robust than free text from a small model."""
    import math
    if not ensure_server():
        raise RuntimeError("the local model server could not be started")
    content = _build_content(prompt + "\nAnswer with the letter only.", images)
    body = {"messages": [{"role": "user", "content": content}], "temperature": 0.0, "max_tokens": 1,
            "logprobs": True, "top_logprobs": 20}
    r = requests.post(URL + "/v1/chat/completions", json=body, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError("model server error %s: %s" % (r.status_code, r.text[:200]))
    top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    p = {l: 0.0 for l in letters}
    for t in top:
        tok = t["token"].strip().upper().rstrip(")").rstrip(".")
        if tok in p:
            p[tok] += math.exp(t["logprob"])
    tot = sum(p.values())
    return {l: (v / tot if tot > 0 else 1.0 / len(letters)) for l, v in p.items()}


def extract_json(text):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except ValueError:
        return None


def ask_json(prompt, images=None, check=None, retries=3, log=None, **kw):
    """Ask for a JSON object. `check(obj)` returns (clean_obj, error). After `retries` failed attempts returns
    (None, last_error) so that the caller can use a deterministic fallback."""
    err = "no attempt"
    for attempt in range(1, retries + 1):
        try:
            text = chat(prompt if attempt == 1 else prompt + "\n\nYour previous answer was rejected: " + err +
                        "\nAnswer again with ONLY the JSON object.", images=images, **kw)
        except Exception as e:
            err = "LLM call failed: %s" % e
            if log:
                log(err)
            time.sleep(3)
            continue
        obj = extract_json(text)
        if obj is None:
            err = "reply is not valid JSON"
        elif check is not None:
            clean, err2 = check(obj)
            if clean is not None and not err2:
                return clean, None
            err = err2 or "rejected"
        else:
            return obj, None
        if log:
            log("attempt %d rejected: %s" % (attempt, err))
    return None, err
