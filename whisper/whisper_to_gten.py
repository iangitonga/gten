import argparse
import hashlib
import urllib
import warnings
import os
from dataclasses import dataclass 

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


@dataclass
class ModelDimensions: 
    n_mels: int = None
    n_vocab: int =  None
    n_audio_ctx: int = None
    n_audio_state: int = None
    n_audio_head: int = None
    n_audio_layer: int = None
    n_text_ctx: int = None
    n_text_state: int = None
    n_text_head: int = None
    n_text_layer: int = None


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, n_head, n_state):
        super().__init__()
        
        self.n_head = n_head
        self.n_state = n_state
        self.d_head = n_state // n_head
        self.query = nn.Linear(n_state, n_head * self.d_head)
        self.key = nn.Linear(n_state, n_head * self.d_head, bias=False)
        self.value = nn.Linear(n_state, n_head * self.d_head)
        self.out = nn.Linear(n_head * self.d_head, n_state)
    

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, n_head, n_state):
        super().__init__()
        
        self.n_head = n_head
        self.n_state = n_state
        self.d_head = n_state // n_head
        self.query = nn.Linear(n_state, n_head * self.d_head)
        self.key = nn.Linear(n_state, n_head * self.d_head, bias=False)
        self.value = nn.Linear(n_state, n_head * self.d_head)
        self.out = nn.Linear(n_head * self.d_head, n_state)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, n_state, n_head, n_mlp, cross_attention=False):
        super().__init__()
        
        self.attn = MultiHeadSelfAttention(n_head, n_state)
        self.attn_ln = nn.LayerNorm(n_state)

        self.cross_attn = MultiHeadCrossAttention(n_head, n_state) if cross_attention else None
        self.cross_attn_ln = nn.LayerNorm(n_state) if cross_attention else None

        self.mlp = nn.Sequential(nn.Linear(n_state, n_mlp), nn.GELU(), nn.Linear(n_mlp, n_state))
        self.mlp_ln = nn.LayerNorm(n_state)

    def forward(self, x, xa=None, mask=None, cache=None):
        x = x + self.attn(self.attn_ln(x), mask=mask)
        if self.cross_attn:
            x = x + self.cross_attn(self.cross_attn_ln(x), xa, cache)
        x = x + self.mlp(self.mlp_ln(x))
        return x
    
    
class AudioEncoder(nn.Module):
    def __init__(self, n_mels, n_audio_layer, n_audio_ctx, n_audio_state, n_audio_head):
        super().__init__()
        
        self.conv1 = nn.Conv1d(n_mels, n_audio_state, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(n_audio_state, n_audio_state, kernel_size=3, stride=2, padding=1)
        self.gelu = nn.GELU()
        self.register_buffer("positional_embedding", self._get_pos_encoding(n_audio_ctx, n_audio_state), persistent=True)

        n_audio_mlp = n_audio_state * 4
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_audio_state, n_audio_head, n_audio_mlp) for _ in range(n_audio_layer)]
        )
        self.ln_post = nn.LayerNorm(n_audio_state)
    
    def _get_pos_encoding(self, n_audio_ctx, n_audio_state):
        dim_mask = torch.arange(n_audio_state//2).view(1, -1)
        factor = torch.log(torch.tensor(10_000)) / (n_audio_state // 2 - 1)
        dim_mask = torch.exp(-factor * dim_mask)
        pos_mask = torch.arange(n_audio_ctx).view(n_audio_ctx, 1)
        mask = pos_mask * dim_mask
        pos_encoding = torch.cat((torch.sin(mask), torch.cos(mask)), dim=1)
        return pos_encoding
    

class TextDecoder(nn.Module):
    def __init__(self, n_vocab, n_text_layer, n_text_ctx, n_text_state, n_text_head):
        super().__init__()
        
        self.token_embedding = nn.Embedding(n_vocab, n_text_state)
        self.positional_embedding = nn.Parameter(torch.empty(n_text_ctx, n_text_state))
        n_text_mlp = n_text_state * 4
        self.blocks = nn.ModuleList(
            [ResidualAttentionBlock(n_text_state, n_text_head, n_text_mlp, cross_attention=True)
            for _ in range(n_text_layer)]
        )
        self.ln = nn.LayerNorm(n_text_state)

        mask = torch.full((n_text_ctx, n_text_ctx), float("-Infinity")).triu_(diagonal=1)
        self.register_buffer("mask", mask)
    

class Whisper(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.dims = dims

        self.encoder = AudioEncoder(
            n_mels=dims.n_mels,
            n_audio_layer=dims.n_audio_layer,
            n_audio_ctx=dims.n_audio_ctx, 
            n_audio_state=dims.n_audio_state,
            n_audio_head=dims.n_audio_head, 
        )
        self.decoder = TextDecoder(
            n_vocab=dims.n_vocab, 
            n_text_layer=dims.n_text_layer, 
            n_text_ctx=dims.n_text_ctx,
            n_text_state=dims.n_text_state, 
            n_text_head=dims.n_text_head,
        )


def save_weight(f, w, name):
    w = w.detach().flatten().numpy().astype(np.float16)
    bytes_ = w.tobytes()
    name = name.encode()
    f.write(int.to_bytes(len(name), 4, "little"))
    f.write(name)
    f.write(int.to_bytes(len(bytes_), 4, "little"))
    f.write(bytes_)

def int_to_bytes(integer, width=4):
    return int.to_bytes(integer, width, "little")

def write_config(f, dims):
    f.write(int_to_bytes(dims["n_mels"]))
    f.write(int_to_bytes(dims["n_vocab"]))
    f.write(int_to_bytes(dims["n_audio_ctx"]))
    f.write(int_to_bytes(dims["n_audio_state"]))
    f.write(int_to_bytes(dims["n_audio_head"]))
    f.write(int_to_bytes(dims["n_audio_layer"]))
    f.write(int_to_bytes(dims["n_text_ctx"]))
    f.write(int_to_bytes(dims["n_text_state"]))
    f.write(int_to_bytes(dims["n_text_head"]))
    f.write(int_to_bytes(dims["n_text_layer"]))
    
def write_encoder(f, model):
    e = model.encoder
    save_weight(f, e.conv1.weight, "enc.conv1.w")
    save_weight(f, e.conv1.bias, "enc.conv1.b")
    save_weight(f, e.conv2.weight, "enc.conv2.w")
    save_weight(f, e.conv2.bias, "enc.conv2.b")
    save_weight(f, e.positional_embedding, "enc.pos_emb.w")
    
    for i in range(len(e.blocks)):
        block = e.blocks[i]
        block_name = f"enc.b{i}"
        save_weight(f, block.attn.query.weight, f"{block_name}.attn.query.w")
        save_weight(f, block.attn.query.bias, f"{block_name}.attn.query.b")
        save_weight(f, block.attn.key.weight, f"{block_name}.attn.key.w")
        save_weight(f, torch.zeros(block.attn.key.weight.shape[0]), f"{block_name}.attn.key.b")
        save_weight(f, block.attn.value.weight, f"{block_name}.attn.value.w")
        save_weight(f, block.attn.value.bias, f"{block_name}.attn.value.b")
        save_weight(f, block.attn.out.weight, f"{block_name}.attn.qkv_proj.w")
        save_weight(f, block.attn.out.bias, f"{block_name}.attn.qkv_proj.b")
        save_weight(f, block.attn_ln.weight, f"{block_name}.attn_ln.w")
        save_weight(f, block.attn_ln.bias, f"{block_name}.attn_ln.b")
        save_weight(f, block.mlp[0].weight, f"{block_name}.mlp_fc.w")
        save_weight(f, block.mlp[0].bias, f"{block_name}.mlp_fc.b")
        save_weight(f, block.mlp[2].weight, f"{block_name}.mlp_proj.w")
        save_weight(f, block.mlp[2].bias, f"{block_name}.mlp_proj.b")
        save_weight(f, block.mlp_ln.weight, f"{block_name}.mlp_ln.w")
        save_weight(f, block.mlp_ln.bias, f"{block_name}.mlp_ln.b")
        
    
    save_weight(f, e.ln_post.weight, "enc.ln.w")
    save_weight(f, e.ln_post.bias, "enc.ln.b")
    

def write_decoder(f, model):
    d = model.decoder
    save_weight(f, d.token_embedding.weight, "dec.emb.w")
    save_weight(f, d.positional_embedding, "dec.pos_emb.w")
    
    for i in range(len(d.blocks)):
        b = model.decoder.blocks[i]
        block_name = f"dec.b{i}"
        save_weight(f, b.attn.query.weight, f"{block_name}.attn.query.w")
        save_weight(f, b.attn.query.bias, f"{block_name}.attn.query.b")
        save_weight(f, b.attn.key.weight, f"{block_name}.attn.key.w")
        save_weight(f, torch.zeros(b.attn.key.weight.shape[0]), f"{block_name}.attn.key.b")
        save_weight(f, b.attn.value.weight, f"{block_name}.attn.value.w")
        save_weight(f, b.attn.value.bias, f"{block_name}.attn.value.b")
        save_weight(f, b.attn.out.weight, f"{block_name}.attn.qkv_proj.w")
        save_weight(f, b.attn.out.bias, f"{block_name}.attn.qkv_proj.b")

        save_weight(f, b.attn_ln.weight, f"{block_name}.attn_ln.w")
        save_weight(f, b.attn_ln.bias, f"{block_name}.attn_ln.b")

        save_weight(f, b.cross_attn.query.weight, f"{block_name}.cross_attn.query.w")
        save_weight(f, b.cross_attn.query.bias, f"{block_name}.cross_attn.query.b")
        save_weight(f, b.cross_attn.key.weight, f"{block_name}.cross_attn.key.w")
        save_weight(f, torch.zeros(b.cross_attn.key.weight.shape[0]), f"{block_name}.cross_attn.key.b")
        save_weight(f, b.cross_attn.value.weight, f"{block_name}.cross_attn.value.w")
        save_weight(f, b.cross_attn.value.bias, f"{block_name}.cross_attn.value.b")
        save_weight(f, b.cross_attn.out.weight, f"{block_name}.cross_attn.qkv_proj.w")
        save_weight(f, b.cross_attn.out.bias, f"{block_name}.cross_attn.qkv_proj.b")

        save_weight(f, b.cross_attn_ln.weight, f"{block_name}.cross_attn_ln.w")
        save_weight(f, b.cross_attn_ln.bias, f"{block_name}.cross_attn_ln.b")

        save_weight(f, b.mlp[0].weight, f"{block_name}.mlp_fc.w")
        save_weight(f, b.mlp[0].bias, f"{block_name}.mlp_fc.b")
        save_weight(f, b.mlp[2].weight, f"{block_name}.mlp_proj.w")
        save_weight(f, b.mlp[2].bias, f"{block_name}.mlp_proj.b")
        save_weight(f, b.mlp_ln.weight, f"{block_name}.mlp_ln.w")
        save_weight(f, b.mlp_ln.bias, f"{block_name}.mlp_ln.b")
    
    save_weight(f, d.ln.weight, "dec.ln.w")
    save_weight(f, d.ln.bias, "dec.ln.b")


GTEN_MAGIC_NUMBER = 0x454c49464e455447
MODELS_URLS = {
    "tiny.en": "https://openaipublic.azureedge.net/main/whisper/models/d3dd57d32accea0b295c96e26691aa14d8822fac7d9d27d5dc00b4ca2826dd03/tiny.en.pt",
    "base.en": "https://openaipublic.azureedge.net/main/whisper/models/25a8566e1d0c1e2231d1c762132cd20e0f96a85d16145c3a00adf5d1ac670ead/base.en.pt",
    "small.en": "https://openaipublic.azureedge.net/main/whisper/models/f953ad0fd29cacd07d5a9eda5624af0f6bcf2258be67c92b79389873d91e0872/small.en.pt",
    "medium.en": "https://openaipublic.azureedge.net/main/whisper/models/d7440d1dc186f76616474e0ff0b3b6b879abc9d1a4926b7adfa41db2d497ab4f/medium.en.pt",
}


def download_model(url: str, root: str) -> str:
    os.makedirs(root, exist_ok=True)

    expected_sha256 = url.split("/")[-2]
    download_target = os.path.join(root, os.path.basename(url))

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        model_bytes = open(download_target, "rb").read()
        if hashlib.sha256(model_bytes).hexdigest() == expected_sha256:
            return download_target
        else:
            warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True, unit_divisor=1024) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    model_bytes = open(download_target, "rb").read()
    if hashlib.sha256(model_bytes).hexdigest() != expected_sha256:
        raise RuntimeError("Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model.")

    return download_target


def convert_model(model_name, model_path):
    if not model_path:
        model_path = download_model(MODELS_URLS[model_name], "downloaded")
    print(f"Converting: {model_path}")

    with open(model_path, "rb") as fp:
        checkpoint = torch.load(fp, map_location="cpu")

    model = Whisper(ModelDimensions(**checkpoint["dims"]))
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    os.makedirs("converted", exist_ok=True)
    with open(f"converted/whisper.{model_name}.gten", "wb") as f:
        f.write(int_to_bytes(GTEN_MAGIC_NUMBER, width=8))
        dims = checkpoint["dims"]
        write_config(f, dims)
        print("Converting encoder...")
        write_encoder(f, model)
        print("Converting decoder...")
        write_decoder(f, model)
    print("Conversion complete.")

parser = argparse.ArgumentParser()
parser.add_argument("model", help="Model name to be converted.", choices=MODELS_URLS.keys())
parser.add_argument("--path", default="", help="Optional path to source model if you have it locally.")

args = parser.parse_args()

convert_model(args.model, args.path)
