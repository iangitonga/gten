#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS = ("Gpt2", "Gpt2-medium", "Gpt2-large")

MODELS_URLS = {
    "Gpt2.fp32": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2.fp32.gten",
    "Gpt2.fp16": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2.fp16.gten",
    "Gpt2-medium.fp32": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-medium.fp32.gten",
    "Gpt2-medium.fp16": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-medium.fp16.gten",
    "Gpt2-large.fp32": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-large.fp32.gten",
    "Gpt2-large.fp16": "https://huggingface.co/iangitonga/gten/resolve/main/Gpt2-large.fp16.gten",
}


def show_progress(cur_size, max_size):
    ls = [" "] * 50
    prog = int(cur_size / max_size * 50)
    for i in range(prog):
        ls[i] = "#"
    print("Progress: [" + "".join(ls) + "]", end="\r", flush=True)
    if cur_size == max_size:
        print()

def _download_model(url, model_path):
    print("Downloading ...")
    with request.urlopen(url) as source, open(model_path, "wb") as output:
        download_size = int(source.info().get("Content-Length"))
        while True:
            buffer = source.read(8192)
            if not buffer:
                break

            output.write(buffer)
            show_progress(len(buffer), download_size)

def download_model(model_name, inference_mode):
    if inference_mode == "f16":
        model_path = os.path.join("models", f"{model_name}.fp16.gten")
    elif inference_mode == "q8":
        model_path = os.path.join("models", f"{model_name}.q8.gten")
    else:
        model_path = os.path.join("models", f"{model_name}.fp32.gten")
    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    # TODO: ADD Q8 URL.
    model_url_key = f"{model_name}.fp16" if inference_mode == "f16" else f"{model_name}.fp32"
    _download_model(MODELS_URLS[model_url_key], model_path)


if len(sys.argv) < 3 or sys.argv[1] not in MODELS:
    print("Model not specified.\n")
    print("usage: model_registry.py MODEL INFERENCE_MODE")
    print("MODEL is one of (Gpt2, Gpt2-medium, Gpt2-large)")
    print("INFERENCE_MODE is one of (f16, f32, q8)")


try:
    download_model(sys.argv[1], sys.argv[2])
except:
    exit(-2)
