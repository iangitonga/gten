#!/usr/bin/python3

import os
import sys
from urllib import request

MODELS = {
    "GPT-2-117M": "https://huggingface.co/iangitonga/gten/resolve/main/GPT-2-117M.gten",
    "GPT-2-345M": "https://huggingface.co/iangitonga/gten/resolve/main/GPT-2-345M.gten",
    "GPT-2-762M": "https://huggingface.co/iangitonga/gten/resolve/main/GPT-2-762M.gten",
}


def show_progress(cur_size, max_size):
    ls = [" "] * 50
    prog = int(cur_size / max_size * 50)
    for i in range(prog):
        ls[i] = "#"
    print("Progress: [" + "".join(ls) + "]", end="\r")
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
    model_path = os.path.join("models", f"{model_name}.gten")
    if os.path.exists(model_path):
        return
    os.makedirs("models", exist_ok=True)
    _download_model(MODELS[model_name], model_path)


if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
    print("Model not specified.\n")
    print("usage: model_registry.py MODEL")
    print("MODEL is one of (GPT-2-117M, GPT-2-345M, GPT-2-762M)")


try:
    download_model(sys.argv[1])
except:
    exit(-2)
