import argparse
import array
import math
import os
import urllib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


# MODEL CONFIG PARAMETERS
@dataclass
class ModelConfig: 
    n_vocab: int =  None
    n_ctx: int = None
    n_state: int = None
    n_layer: int = None
    n_mlp: int = None
    n_head: int = None



class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head: int, n_state: int, n_ctx: int):
        super().__init__()
        
        self.n_head = n_head
        self.n_state = n_state
        self.c_attn = nn.Linear(n_state, n_state * 3)
        self.c_proj = nn.Linear(n_state, n_state)
        
        # The masking attn mask.
        bias = torch.empty((1, 1, n_ctx, n_ctx))
        self.register_buffer('bias', bias, persistent=True)
        
    def forward(self, x):
        q, k, v = self.c_attn(x).split(self.n_state, dim=2)
        qkv = self._qkv_attention(q, k, v)
        out = self.c_proj(qkv)
        return out
    
    def _qkv_attention(self, q, k, v):
        n_batch, n_ctx = q.shape[0], q.shape[1]
        d_head = self.n_state // self.n_head
        q = q.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 1, 3)
        k = k.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 3, 1)
        v = v.view(n_batch, n_ctx, self.n_head, d_head).permute(0, 2, 1, 3)
        scale = 1.0 / math.sqrt(d_head)
        qk = (q @ k) * scale
        qk = qk.masked_fill(self.bias[:, :, :n_ctx, :n_ctx] == 0, float('-inf'))
        qk = F.softmax(qk, dim=-1)
        qkv = qk @ v
        qkv = qkv.permute(0, 2, 1, 3).flatten(start_dim=2)
        return qkv
    

class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state: int, n_head: int, n_mlp: int, n_ctx: int):
        super().__init__()
        
        self.attn = MultiHeadSelfAttention(n_head, n_state, n_ctx)
        self.ln_1 = nn.LayerNorm(n_state)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(n_state, n_mlp),
            c_proj  = nn.Linear(n_mlp, n_state),
            act     = nn.GELU(approximate="tanh"),
        ))
        self.mlpf = lambda x: self.mlp.c_proj(self.mlp.act(self.mlp.c_fc(x))) # MLP forward
        self.ln_2 = nn.LayerNorm(n_state)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlpf(self.ln_2(x))
        return x
    

# For inference only. no dropout layers.
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.wte = nn.Embedding(config.n_vocab, config.n_state)
        self.wpe = nn.Embedding(config.n_ctx, config.n_state)
        self.h = nn.ModuleList([
            ResidualAttentionBlock(config.n_state, config.n_head, config.n_mlp, config.n_ctx) for _ in range(config.n_layer)
        ])
        self.ln_f = nn.LayerNorm(config.n_state)
        
    def forward(self, x):
        pos = torch.arange(0, x.shape[1], dtype=torch.long).unsqueeze(0)
        x = self.wte(x) + self.wpe(pos)
        for block in self.h:
            x = block(x)
        x = self.ln_f(x)
        logits = (x @ torch.transpose(self.wte.weight.to(x.dtype), 0, 1)).float()
        return logits

    @classmethod
    def from_pretrained(cls, path, config):
        model = cls(config)
        gpt_state = torch.load(path, map_location="cpu")
        for key in gpt_state.keys():
            if (key.endswith("attn.c_attn.weight")
                or key.endswith("attn.c_proj.weight")
                or key.endswith("mlp.c_fc.weight")
                or key.endswith("mlp.c_proj.weight")):
                gpt_state[key] = gpt_state[key].transpose(0, 1)
        model.load_state_dict(gpt_state)
        return model


GTEN_MAGIC_NUMBER = 0x454c49464e455447
BYTEORDER = "little"


def itob(integer, width=4):
    return int.to_bytes(integer, width, BYTEORDER)

def btoi(bytes_):
    return int.from_bytes(bytes_, BYTEORDER)


class Qparams:
    def __init__(self, scale: float, zero_point: int):
        self.s_bytes = np.array([scale], dtype=np.float32).tobytes()
        self.z_bytes = np.array([zero_point], dtype=np.int32).tobytes()

DEFAULT_QPARAMS = Qparams(0.0, 0)

def quantize_weight(weight):
    aq, bq = -128, 127
    a, b = weight.min(), weight.max()
    scale = (b-a)/(bq-aq)
    zero_point = int((b*aq - a*bq)/(b-a))
    qtensor = np.round(weight * 1/scale) + zero_point
    qtensor = np.clip(qtensor, a_min=aq, a_max=bq).astype(np.int8)
    qparams = Qparams(scale, zero_point)
    return qtensor, qparams


def write_layer(fout, name, activ_size, activ_qparams, weights_size, w0, bias):
    name = name.encode()
    # <layer_name_size, layer_name>
    fout.write(itob(len(name)))
    fout.write(name)

    # Activations
    # <act_name_size, act_qs, act_qz>
    fout.write(itob(activ_size))
    fout.write(activ_qparams.s_bytes)
    fout.write(activ_qparams.z_bytes)

    # W0
    if w0 is not None:
        w0_name = f"{name.decode()}.weight".encode()
        fout.write(itob(len(w0_name)))
        fout.write(w0_name)
        fout.write(itob(weights_size))
        w0 = w0.detach().numpy().flatten()
        w0_qparams = DEFAULT_QPARAMS
        if weights_size == 1:
            w0, w0_qparams = quantize_weight(w0)
        elif weights_size == 2:
            w0 = w0.astype(np.float16)
        else:
            w0 = w0.astype(np.float32)
        fout.write(w0_qparams.s_bytes)
        fout.write(w0_qparams.z_bytes)
        w0_bytes = w0.tobytes()
        fout.write(itob(len(w0_bytes)))
        fout.write(w0_bytes)

    # Bias
    if bias is not None:
        # NOTE: We do not quantize biases.
        bias_name = f"{name.decode()}.bias".encode()
        fout.write(itob(len(bias_name)))
        fout.write(bias_name)
        bias_size = 2 if weights_size == 2 else 4
        fout.write(itob(bias_size))

        bias = bias.detach().numpy().flatten()
        bias_qparams = DEFAULT_QPARAMS
        if weights_size == 2:
            bias = bias.astype(np.float16)
        else:
            bias = bias.astype(np.float32)

        fout.write(bias_qparams.s_bytes)
        fout.write(bias_qparams.z_bytes)
        bias_bytes = bias.tobytes()
        fout.write(itob(len(bias_bytes)))
        fout.write(bias_bytes)


def write_block(fout, name, model, weights_size, activ_size, block_idx):
    name = name.encode()
    # <block_name_size, block_name>
    fout.write(itob(len(name)))
    fout.write(name)

    h = model.h[block_idx]
    attn_qw, attn_kw, attn_vw = h.attn.c_attn.weight.split(model.config.n_state, dim=0)
    attn_qb, attn_kb, attn_vb = h.attn.c_attn.bias.split(model.config.n_state, dim=0)

    layer_name = f"{name.decode()}.attn.q"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=weights_size, w0=attn_qw, bias=attn_qb)

    layer_name = f"{name.decode()}.attn.k"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=weights_size, w0=attn_kw, bias=attn_kb)

    layer_name = f"{name.decode()}.attn.v"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=weights_size, w0=attn_vw, bias=attn_vb)

    attn_pw = h.attn.c_proj.weight
    attn_pb = h.attn.c_proj.bias
    layer_name = f"{name.decode()}.attn.c_proj"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=weights_size, w0=attn_pw, bias=attn_pb)

    ln_1w = h.ln_1.weight
    ln_1b = h.ln_1.bias
    layer_name = f"{name.decode()}.ln_1"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=weights_size, w0=ln_1w, bias=ln_1b)

    mlp_fcw = h.mlp.c_fc.weight
    mlp_fcb = h.mlp.c_fc.bias
    layer_name = f"{name.decode()}.mlp.c_fc"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=weights_size, w0=mlp_fcw, bias=mlp_fcb)

    mlp_pw = h.mlp.c_proj.weight
    mlp_pb = h.mlp.c_proj.bias
    layer_name = f"{name.decode()}.mlp.c_proj"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=weights_size, w0=mlp_pw, bias=mlp_pb)

    ln_2w = h.ln_2.weight
    ln_2b = h.ln_2.bias
    layer_name = f"{name.decode()}.ln_2"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=weights_size, w0=ln_2w, bias=ln_2b)

    layer_name = f"{name.decode()}.attn.qkv_out"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=None, w0=None, bias=None)

    layer_name = f"{name.decode()}.mlp.gelu"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=None, w0=None, bias=None)

    layer_name = f"{name.decode()}.inp_res"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=None, w0=None, bias=None)

    layer_name = f"{name.decode()}.attn_res"
    write_layer(
        fout, layer_name, activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
        weights_size=None, w0=None, bias=None)

def convert_model_to_gten(model, model_name, vocab_fname, weights_size):
    if weights_size == 1:
        model_fname = f"{model_name}.q8.gten"
        activ_size = 4
    elif weights_size == 2:
        model_fname = f"{model_name}.fp16.gten"
        activ_size = 2
    else:
        model_fname = f"{model_name}.fp32.gten"
        activ_size = 4


    with open(model_fname, "wb") as fout:
        fout.write(itob(GTEN_MAGIC_NUMBER, width=8))
        fout.write(itob(model.config.n_vocab))
        fout.write(itob(model.config.n_ctx))
        fout.write(itob(model.config.n_state))
        fout.write(itob(model.config.n_layer))
        fout.write(itob(model.config.n_head))
        
        print("Writing vocab")
        segment_name = b"vocab"
        segment_name_size = len(segment_name)
        expected_size = 521_855
        with open(vocab_fname, "rb") as vf:
            vocab_bytes = vf.read()
            segment_size = len(vocab_bytes)
            assert segment_size == expected_size, f"Vocab: expected={expected_size}, real={segment_size}"
            fout.write(itob(segment_name_size))
            fout.write(segment_name)
            fout.write(itob(segment_size))
            fout.write(vocab_bytes)
        
        print("Converting wte")
        write_layer(
            fout, "wte", activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
            weights_size=weights_size, w0=model.wte.weight, bias=None)
        
        print("Converting wpe")
        write_layer(
            fout, "wpe", activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
            weights_size=weights_size, w0=model.wpe.weight, bias=None)

        write_layer(
            fout, "emb_res", activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
            weights_size=None, w0=None, bias=None)
            
        for i in range(model.config.n_layer):
            print(f"Converting block_{i}")
            write_block(fout, f"h.{i}", model, weights_size, activ_size, i)
        
        print("Converting ln_f")
        write_layer(
            fout, "ln_f", activ_size=activ_size, activ_qparams=DEFAULT_QPARAMS,
            weights_size=weights_size, w0=model.ln_f.weight, bias=model.ln_f.bias)



# Upload vocab fname[down].
# Download model.

def download_model(model_name, url):
    model_path = f"{model_name}"

    print(f"Downloading {model_name}")
    with urllib.request.urlopen(url) as source, open(model_path, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))
    return model_path

def download_vocab(url):
    print("Downloading vocab")
    vocab_path = "gpt2_vocab.bin"
    with urllib.request.urlopen(url) as source, open(vocab_path, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    return vocab_path


VOCAB_URL = "https://huggingface.co/iangitonga/gten/resolve/main/gpt-2-vocab.gten"

MODEL_URL = {
    "Gpt2": "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin",
    "Gpt2-medium": "https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin",
    "Gpt2-large": "https://huggingface.co/gpt2-large/resolve/main/pytorch_model.bin",
    "Gpt2-xl": "https://huggingface.co/gpt2-xl/resolve/main/pytorch_model.bin",
}

MODEL_CONFIG = {
    "Gpt2":        ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state =  768, n_layer = 12, n_head = 12, n_mlp =  768 * 4),
    "Gpt2-medium": ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1024, n_layer = 24, n_head = 16, n_mlp = 1024 * 4),
    "Gpt2-large":  ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1280, n_layer = 36, n_head = 20, n_mlp = 1280 * 4),
    "Gpt2-xl":     ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1600, n_layer = 48, n_head = 25, n_mlp = 1600 * 4),
}


def download_and_convert(model_name, dtype, model_path, vocab_path):
    if dtype == "qwi8":
        weights_size = 1
    elif dtype == "fp16":
        weights_size = 2
    elif dtype == "fp32":
        weights_size = 4
    else:
        assert False

    print(F"Converting to dtype: {dtype}")

    if not model_path:
        model_path = download_model(model_name, MODEL_URL[model_name])
    if not vocab_path:
        vocab_path = download_vocab(VOCAB_URL)
    model = Transformer.from_pretrained(model_path, MODEL_CONFIG[model_name])
    convert_model_to_gten(model, model_name, vocab_path, weights_size)
    print("Conversion complete!!!")


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model name to be converted.", choices=MODEL_CONFIG.keys())
parser.add_argument("dtype", help="Weights dtype.", choices=("fp32", "fp16", "qwi8"))
parser.add_argument("--mpath", help="Optional path to source model if you have it locally.")
parser.add_argument("--vpath", help="Optional path to vocab if you have it locally.")

args = parser.parse_args()

download_and_convert(args.model, args.dtype, args.mpath, args.vpath)

        
"""

OPTIONS:
~ gpt_to_gten.py MODEL DTYPE OPTIONS

MODEL: {Gpt2, Gpt2-medium, Gpt2-large, Gpt2-xl}
DTYPE: {f32, f16, qwi8}
OPTIONS:
    
------------------
 Activ   | Weight
 --------|--------
[Float32, Float32]  ~fpcvt
[Float16, Float16]  ~fpcvt
[  Qint8,   Qint8]  --full-quantization 
[Float32,   Qint8]  --weight-only-quantization
[Qint8,   Float32]  --activation-only-quantization

~ qcvt<overload_t>(inp):      FP32 <-> FP16 conv: template
~ dequantize(overload_t inp): overload for {FP32, Qint}
"""