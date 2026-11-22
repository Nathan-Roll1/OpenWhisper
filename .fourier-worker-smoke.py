"""Exercise the unmodified OpenWhisper speech process, using a public recording."""
import array
import hashlib
import json
import os
import re
import sys
import threading
import urllib.request
import wave
from pathlib import Path

from services import components
from services.local_asr.process import SpeechProcess

pin = "8bcce21695e4ba13569c48de6cedffcb8a7576f5"
root = Path.cwd() / "fourier-smoke"
root.mkdir(exist_ok=True)
weights = root / "Fourier-2B-Q8_0.gguf"
release = f"https://huggingface.co/oruk/oruk-fourier-2b/resolve/{pin}/"
cancel = threading.Event()
def progress(*_args): pass
component = components.ComponentId.ASR_TRANSCRIBE
spec = components.catalog_entry_for_platform(component)
assert spec["version"].startswith("0.4.1")
components.install_component(component, spec, progress, cancel)
runtime = Path(components.active_component_dir(component))
ready = components.is_installed(component)
assert ready
components._download_verified(release + weights.name, "6a7c419bd571f7b0e21da2d5ab1e3a1a3938a83470d622be57427af98f322981",
                              2185030848, str(weights), progress, cancel)
url = "https://raw.githubusercontent.com/ggml-org/whisper.cpp/b0a11594aec50892a02cd8d129eee2dfe93a8bb8/samples/jfk.wav"
raw = urllib.request.urlopen(url, timeout=30).read()
assert hashlib.sha256(raw).hexdigest() == "59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e"
clip = root / "jfk.wav"
clip.write_bytes(raw)
with wave.open(str(clip), "rb") as wav:
    assert (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) == (16000, 1, 2)
    audio = array.array("h", wav.readframes(wav.getnframes()))
file = root / "input.f32"
file.write_bytes(array.array("f", (v / 32768 for v in audio)).tobytes())
python = str(runtime / "python.exe") if sys.platform == "win32" else sys.executable
ref = "and so my fellow americans ask not what your country can do for you ask what you can do for your country".split()
for i in range(2):
    process = SpeechProcess(python)
    try:
        result = process.request("load", backend="qwen_asr", model="fourier-2b", model_path=str(weights),
                                 runtime=str(runtime), device="cpu", timeout=300)
        assert result["device"] == "cpu"
        decoded = process.request("transcribe", audio_path=str(file), language="en", timeout=200)
        words = re.findall("[a-z]+", decoded["text"].lower())
        assert words == ref, (len(words), len(ref), hashlib.sha256(decoded["text"].encode()).hexdigest())
        assert decoded["segments"]
        print(json.dumps({"run": i+1, "platform":components.current_platform_tag(), "words": len(ref),
                          "exact_normalized": True, "device": result["device"], "segments":len(decoded["segments"]),
                          "runtime_version":spec["version"], "worker_python":Path(python).name}),flush=True)
    finally:
        process.close()
print("OPENWHISPER_PRODUCTION_WORKER_PASSED")
