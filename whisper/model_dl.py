#!/usr/bin/python3

import os
import sys
from urllib import request


MODELS_URLS = {
    "tiny": "https://huggingface.co/iangitonga/gten/resolve/main/whisper.tiny.en.gten",
    "base": "https://huggingface.co/iangitonga/gten/resolve/main/whisper.base.en.gten",
    "small": "https://huggingface.co/iangitonga/gten/resolve/main/whisper.small.en.gten",
    "medium": "https://huggingface.co/iangitonga/gten/resolve/main/whisper.medium.en.gten"
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

def download_model(model_name):
    model_path = os.path.join("models", f"whisper.{model_name}.en.gten")
    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    _download_model(MODELS_URLS[model_name], model_path)


if len(sys.argv) < 2 or sys.argv[1] not in MODELS_URLS.keys():
    print("Model not specified.\n")
    print("usage: model_dl.py MODEL")
    print("MODEL is one of (tiny, base, small, medium)")


try:
    download_model(sys.argv[1])
except:
    exit(-2)
