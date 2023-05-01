import argparse
import array
import math
import os
import urllib
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
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


def itob(integer):
    return int.to_bytes(integer, 8, BYTEORDER)

def btoi(bytes_):
    return int.from_bytes(bytes_, BYTEORDER)

def tensor_to_file(tensor, file_obj, fpsize, expected_size):
    assert (fpsize == 2 or fpsize == 4), f"Unsupported datasize: {fpsize}"
    tensor = tensor.flatten().to(torch.float16) if fpsize == 2 else tensor.flatten().to(torch.float32)
    # A list of bytes as integers.
    tensor_bytes = tensor.detach().numpy().tobytes()
    assert len(tensor_bytes) == expected_size
    file_obj.write(itob(len(tensor_bytes)))
    # byteorder issue?
    file_obj.write(tensor_bytes)


def test_converted(fname, model, dtype_size):
    print(f"Testing conversion of: {fname}")
    with open(fname, "rb") as f:
        magic = f.read(8)
        assert btoi(magic) == GTEN_MAGIC_NUMBER
        _dtype_size = f.read(8)
        assert btoi(_dtype_size) == dtype_size
        n_vocab = f.read(8)
        assert btoi(n_vocab) == model.config.n_vocab
        n_ctx = f.read(8)
        assert btoi(n_ctx) == model.config.n_ctx
        n_state = f.read(8)
        assert btoi(n_state) == model.config.n_state
        n_layer = f.read(8)
        assert btoi(n_layer) == model.config.n_layer
        n_head = f.read(8)
        assert btoi(n_head) == model.config.n_head

        block_names = [f"block_{i}".encode() for i in range(model.config.n_layer)]
        segment_names = [b"vocab", b"wte", b"wpe", *block_names, b"ln_f"]

        for segment_name in segment_names:
            segment_name_size = btoi(f.read(8))
            assert segment_name_size == len(segment_name)
            assert f.read(segment_name_size) == segment_name
            segment_size = btoi(f.read(8))
            f.seek(segment_size, os.SEEK_CUR)
    print("Testing successfull!!!")


def write_block(file_obj, model, weights_dtype, dtype_size, block_idx):
    segment_name = f"block_{block_idx}".encode()
    segment_name_size = len(segment_name)
    file_obj.write(itob(segment_name_size))
    file_obj.write(segment_name)

    h = model.h[block_idx]
    attn_qw, attn_kw, attn_vw = h.attn.c_attn.weight.split(model.config.n_state, dim=0)
    attn_qb, attn_kb, attn_vb = h.attn.c_attn.bias.split(model.config.n_state, dim=0)
    attn_pw = h.attn.c_proj.weight
    attn_pb = h.attn.c_proj.bias
    ln_1w = h.ln_1.weight
    ln_1b = h.ln_1.bias
    mlp_fcw = h.mlp.c_fc.weight
    mlp_fcb = h.mlp.c_fc.bias
    mlp_pw = h.mlp.c_proj.weight
    mlp_pb = h.mlp.c_proj.bias
    ln_2w = h.ln_2.weight
    ln_2b = h.ln_2.bias
    block_tensors = [attn_qw, attn_qb, attn_kw, attn_kb, attn_vw, attn_vb, attn_pw, attn_pb,
                   ln_1w, ln_1b, mlp_fcw, mlp_fcb, mlp_pw, mlp_pb, ln_2w, ln_2b]
    block_tensors = [t.flatten() for t in block_tensors]
    combined_tensors = torch.cat(block_tensors, dim=0)
    expected_size = (12 * model.config.n_state * model.config.n_state + 13 * model.config.n_state) * dtype_size
    tensor_to_file(combined_tensors, file_obj, dtype_size, expected_size)

def convert_model_to_gten(model, model_name, vocab_fname, weights_dtype):
    DTYPE_SIZE = 2 if weights_dtype == torch.float16 else 4

    model_fname = f"{model_name}.fp16.gten" if weights_dtype == torch.float16 else f"{model_name}.fp32.gten"
    print("Converting ...")
    with open(model_fname, "wb") as f:
        f.write(itob(GTEN_MAGIC_NUMBER))
        f.write(itob(DTYPE_SIZE))
        f.write(itob(model.config.n_vocab))
        f.write(itob(model.config.n_ctx))
        f.write(itob(model.config.n_state))
        f.write(itob(model.config.n_layer))
        f.write(itob(model.config.n_head))
        
        print("Writing vocab")
        segment_name = b"vocab"
        segment_name_size = len(segment_name)
        expected_size = 521_855
        with open(vocab_fname, "rb") as vf:
            vocab_bytes = vf.read()
            segment_size = len(vocab_bytes)
            assert segment_size == expected_size, f"Vocab: expected={expected_size}, real={segment_size}"
            f.write(itob(segment_name_size))
            f.write(segment_name)
            f.write(itob(segment_size))
            f.write(vocab_bytes)
        
        print("Writing wte")
        segment_name = b"wte"
        segment_name_size = len(segment_name)
        f.write(itob(segment_name_size))
        f.write(segment_name)
        expected_size = model.config.n_vocab * model.config.n_state * DTYPE_SIZE
        tensor_to_file(model.wte.weight, f, DTYPE_SIZE, expected_size)
        
        print("Writing wpe")
        segment_name = b"wpe"
        segment_name_size = len(segment_name)
        f.write(itob(segment_name_size))
        f.write(segment_name)
        expected_size = model.config.n_ctx * model.config.n_state * DTYPE_SIZE
        tensor_to_file(model.wpe.weight, f, DTYPE_SIZE, expected_size)
        
        for i in range(model.config.n_layer):
            print(f"Writing block_{i}")
            write_block(f, model, weights_dtype, DTYPE_SIZE, i)
        
        print("Writing ln_f")
        segment_name = b"ln_f"
        segment_name_size = len(segment_name)
        f.write(itob(segment_name_size))
        f.write(segment_name)
        expected_size = model.config.n_state * 2 * DTYPE_SIZE
        combined_data = torch.cat((model.ln_f.weight.flatten(), model.ln_f.bias.flatten()), dim=0)
        tensor_to_file(combined_data, f, DTYPE_SIZE, expected_size)

    # A little sanity check.
    test_converted(model_fname, model, DTYPE_SIZE)


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
    "GPT-2-117M": "https://huggingface.co/gpt2/resolve/main/pytorch_model.bin",
    "GPT-2-345M": "https://huggingface.co/gpt2-medium/resolve/main/pytorch_model.bin",
    "GPT-2-762M": "https://huggingface.co/gpt2-large/resolve/main/pytorch_model.bin",
    # "GPT-2-1542M": "https://huggingface.co/gpt2-xl/resolve/main/pytorch_model.bin",
}

MODEL_CONFIG = {
    "GPT-2-117M": ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state =  768, n_layer = 12, n_head = 12, n_mlp =  768 * 4),
    "GPT-2-345M": ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1024, n_layer = 24, n_head = 12, n_mlp = 1024 * 4),
    "GPT-2-762M": ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1280, n_layer = 36, n_head = 12, n_mlp = 1280 * 4),
    # "GPT-2-1542M":ModelConfig(n_vocab = 50257, n_ctx = 1024, n_state = 1600, n_layer = 48, n_head = 12, n_mlp = 1600 * 4),
}


def download_and_convert(model_name, dtype, model_path, vocab_path):
    weights_dtype = torch.float16 if dtype == "fp16" else torch.float32
    print(F"Converting to dtype: {dtype}[{weights_dtype}]")

    if not model_path:
        model_path = download_model(model_name, MODEL_URL[model_name])
    if not vocab_path:
        vocab_path = download_vocab(VOCAB_URL)
    model = Transformer.from_pretrained(model_path, MODEL_CONFIG[model_name])
    convert_model_to_gten(model, model_name, vocab_path, weights_dtype)
    # os.remove(model_path)
    # os.remove(vocab_path)
    print("Conversion complete!!!")


parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model name to be converted.", choices=MODEL_CONFIG.keys())
parser.add_argument("dtype", help="Weights dtype.", choices=("fp32", "fp16"))
parser.add_argument("--mpath", help="Optional path to source model to avoid download.")
parser.add_argument("--vpath", help="Optional path to vocab to avoid download.")

args = parser.parse_args()

download_and_convert(args.model, args.dtype, args.mpath, args.vpath)
