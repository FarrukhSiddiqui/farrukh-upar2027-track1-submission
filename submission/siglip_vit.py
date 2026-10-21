"""SigLIP2 ViT image encoder + attribute head in plain torch (no timm / open_clip).

Parameter names match timm's VisionTransformer as used by open_clip's
SigLIP2 checkpoints (`visual.trunk.*`), so the converted weights from
make_siglip_init.py load directly: pre-norm blocks, exact GELU, no class
token, final LayerNorm, attention pooling with one learned query ("map").

Like clip_vit.CLIPAttributeNet it takes the pipeline's ImageNet-normalised
input and re-normalises internally (SigLIP uses mean = std = 0.5).
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)

SIGLIP_CONFIGS = {
    # name: (width, layers, heads, patch, pretrain grid side)
    "siglip2_l16": (1024, 24, 16, 16, 16),
}


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(out.transpose(1, 2).reshape(b, n, c))


class Mlp(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim, heads, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, heads)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)
        self.drop_path = drop_path

    def _dp(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        x = x + self._dp(self.attn(self.norm1(x)))
        return x + self._dp(self.mlp(self.norm2(x)))


class AttentionPool(nn.Module):
    """timm AttentionPoolLatent with one latent query, pool='token'."""

    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.latent = nn.Parameter(torch.zeros(1, 1, dim))
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = Mlp(dim, dim * 4)

    def forward(self, x):
        b, n, c = x.shape
        h = self.heads
        q = self.q(self.latent.expand(b, -1, -1)).view(b, 1, h, c // h).transpose(1, 2)
        kv = self.kv(x).view(b, n, 2, h, c // h).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, kv[0], kv[1])
        out = self.proj(out.transpose(1, 2).reshape(b, 1, c))
        out = out + self.mlp(self.norm(out))
        return out[:, 0]


class PatchEmbed(nn.Module):
    def __init__(self, dim, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, dim, patch, patch)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class SigLIPVisionTower(nn.Module):
    def __init__(self, width, layers, heads, patch, grid, drop_path=0.0):
        super().__init__()
        self.grid, self.patch = grid, patch
        self.patch_embed = PatchEmbed(width, patch)
        self.pos_embed = nn.Parameter(torch.zeros(1, grid[0] * grid[1], width))
        rates = [drop_path * i / max(1, layers - 1) for i in range(layers)]
        self.blocks = nn.ModuleList(Block(width, heads, r) for r in rates)
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.attn_pool = AttentionPool(width, heads)

    def forward(self, x):
        size = (self.grid[0] * self.patch, self.grid[1] * self.patch)
        if tuple(x.shape[-2:]) != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
        x = self.patch_embed(x) + self.pos_embed.to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        return self.attn_pool(self.norm(x))


def resize_pos_embed(pos, grid):
    """Bicubic-resize a square [1, g*g, C] table to grid (h, w)."""
    g = int(math.sqrt(pos.shape[1]))
    if (g, g) == tuple(grid):
        return pos
    tok = pos.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
    tok = F.interpolate(tok, size=grid, mode="bicubic", align_corners=False)
    return tok.permute(0, 2, 3, 1).reshape(1, grid[0] * grid[1], -1).to(pos.dtype)


class SigLIPAttributeNet(nn.Module):
    """SigLIP2 image embedding (L2-normalised) -> linear head, text-initialised.

    head.weight starts as scale * (text(positive) - text(negative)) per
    attribute when the init file has text embeddings, i.e. the zero-shot
    classifier; otherwise it starts at zero and the head-only epoch fits it.
    """

    def __init__(self, backbone="siglip2_l16", image_size=(256, 128), num_attrs=40,
                 drop_path=0.1, init_path=None):
        super().__init__()
        width, layers, heads, patch, _ = SIGLIP_CONFIGS[backbone]
        self.backbone = backbone
        grid = (image_size[0] // patch, image_size[1] // patch)
        self.visual = SigLIPVisionTower(width, layers, heads, patch, grid, drop_path)
        self.head = nn.Linear(width, num_attrs)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        a = torch.tensor(IMAGENET_STD) / torch.tensor(SIGLIP_STD)
        b = (torch.tensor(IMAGENET_MEAN) - torch.tensor(SIGLIP_MEAN)) / torch.tensor(SIGLIP_STD)
        self.register_buffer("in_scale", a.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_shift", b.view(1, 3, 1, 1), persistent=False)
        if init_path is not None:
            self.load_init(init_path)

    def load_init(self, path):
        init = torch.load(path, map_location="cpu", weights_only=False)
        visual = dict(init["visual"])
        visual["pos_embed"] = resize_pos_embed(visual["pos_embed"], self.visual.grid)
        self.visual.load_state_dict(visual)
        if "text_pos" in init:
            t_pos = F.normalize(init["text_pos"].float(), dim=-1)
            t_neg = F.normalize(init["text_neg"].float(), dim=-1)
            with torch.no_grad():
                self.head.weight.copy_(init.get("logit_scale", 10.0) * (t_pos - t_neg))

    def forward(self, x):
        x = x * self.in_scale + self.in_shift
        f = F.normalize(self.visual(x).float(), dim=-1)
        return self.head(f)

    def param_groups(self, lr, head_lr, layer_decay, weight_decay):
        """AdamW groups with layer-wise LR decay; no weight decay on norms/biases/embeddings."""
        layers = len(self.visual.blocks)
        groups = {}

        def depth(name):
            if name.startswith("visual.blocks."):
                return int(name.split(".")[2]) + 1
            if name.startswith(("visual.patch_embed", "visual.pos_embed")):
                return 0
            return layers + 1  # final norm, attention pool

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("head."):
                key, g_lr = ("head", 0), head_lr
            else:
                d = depth(name)
                key, g_lr = ("layer", d), lr * layer_decay ** (layers + 1 - d)
            no_wd = p.ndim < 2 or "pos_embed" in name or "latent" in name
            key = key + (no_wd,)
            if key not in groups:
                groups[key] = {"params": [], "lr": g_lr,
                               "weight_decay": 0.0 if no_wd else weight_decay}
            groups[key]["params"].append(p)
        return list(groups.values())
