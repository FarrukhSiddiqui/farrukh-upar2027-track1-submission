"""CLIP ViT image encoder + attribute head in plain torch (no open_clip / timm).

Parameter names match OpenAI's CLIP checkpoints (the `visual.*` part), so the
converted weights from make_clip_init.py load directly. The model takes the
pipeline's ImageNet-normalised input and re-normalises it to CLIP statistics
internally, so the data path is identical for every backbone.

Used by both training (via upar_common) and the submission's run.py.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

CLIP_CONFIGS = {
    # name: (width, layers, heads, patch, embed_dim)
    "clip_vit_b16": (768, 12, 12, 16, 512),
    "clip_vit_l14": (1024, 24, 16, 14, 768),
}


class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    """Same parameters as nn.MultiheadAttention, computed with SDPA."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.in_proj_weight = nn.Parameter(torch.empty(3 * width, width))
        self.in_proj_bias = nn.Parameter(torch.zeros(3 * width))
        self.out_proj = nn.Linear(width, width)

    def forward(self, x):
        b, n, c = x.shape
        qkv = F.linear(x, self.in_proj_weight, self.in_proj_bias)
        qkv = qkv.view(b, n, 3, self.heads, c // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.out_proj(out.transpose(1, 2).reshape(b, n, c))


class Block(nn.Module):
    def __init__(self, width, heads, drop_path=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = Attention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential()
        self.mlp.add_module("c_fc", nn.Linear(width, width * 4))
        self.mlp.add_module("gelu", QuickGELU())
        self.mlp.add_module("c_proj", nn.Linear(width * 4, width))
        self.drop_path = drop_path

    def _dp(self, x):
        if not self.training or self.drop_path == 0:
            return x
        keep = torch.rand(x.shape[0], 1, 1, device=x.device) >= self.drop_path
        return x * keep / (1 - self.drop_path)

    def forward(self, x):
        x = x + self._dp(self.attn(self.ln_1(x)))
        return x + self._dp(self.mlp(self.ln_2(x)))


class Transformer(nn.Module):
    def __init__(self, width, layers, heads, drop_path=0.0):
        super().__init__()
        rates = [drop_path * i / max(1, layers - 1) for i in range(layers)]
        self.resblocks = nn.ModuleList(Block(width, heads, r) for r in rates)

    def forward(self, x):
        for blk in self.resblocks:
            x = blk(x)
        return x


class VisionTransformer(nn.Module):
    def __init__(self, width, layers, heads, patch, embed_dim, grid, drop_path=0.0):
        super().__init__()
        self.grid = grid
        self.conv1 = nn.Conv2d(3, width, patch, patch, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        self.positional_embedding = nn.Parameter(torch.zeros(grid[0] * grid[1] + 1, width))
        self.ln_pre = nn.LayerNorm(width)
        self.transformer = Transformer(width, layers, heads, drop_path)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.zeros(width, embed_dim))

    def forward(self, x):
        # 256x128 is not a multiple of 14 px patches: resize to the patch grid
        # (252x126 for ViT-L/14) instead of letting the conv drop the edges.
        p = self.conv1.kernel_size[0]
        size = (self.grid[0] * p, self.grid[1] * p)
        if tuple(x.shape[-2:]) != size:
            x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
        x = self.conv1(x).flatten(2).transpose(1, 2)
        cls = self.class_embedding.to(x.dtype).expand(x.shape[0], 1, -1)
        x = torch.cat([cls, x], 1) + self.positional_embedding.to(x.dtype)
        x = self.transformer(self.ln_pre(x))
        return self.ln_post(x[:, 0]) @ self.proj


def resize_pos_embed(pos, grid):
    """Bicubic-resize a square CLIP position table [1+g*g, C] to grid (h, w)."""
    cls, tok = pos[:1], pos[1:]
    g = int(math.sqrt(len(tok)))
    if (g, g) == tuple(grid):
        return pos
    tok = tok.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
    tok = F.interpolate(tok, size=grid, mode="bicubic", align_corners=False)
    return torch.cat([cls, tok.permute(0, 2, 3, 1).reshape(grid[0] * grid[1], -1).to(pos.dtype)])


class CLIPAttributeNet(nn.Module):
    """CLIP image embedding (L2-normalised) -> linear head, text-initialised.

    head.weight starts as scale * (text(positive prompt) - text(negative prompt))
    per attribute, i.e. the zero-shot classifier, so fine-tuning starts from
    CLIP's robust solution instead of a random head.
    """

    def __init__(self, backbone="clip_vit_b16", image_size=(256, 128), num_attrs=40,
                 drop_path=0.1, init_path=None):
        super().__init__()
        width, layers, heads, patch, embed = CLIP_CONFIGS[backbone]
        self.backbone = backbone
        grid = (image_size[0] // patch, image_size[1] // patch)
        self.visual = VisionTransformer(width, layers, heads, patch, embed, grid, drop_path)
        self.head = nn.Linear(embed, num_attrs)
        # ImageNet-normalised input -> CLIP-normalised input: x * a + b
        a = torch.tensor(IMAGENET_STD) / torch.tensor(CLIP_STD)
        b = (torch.tensor(IMAGENET_MEAN) - torch.tensor(CLIP_MEAN)) / torch.tensor(CLIP_STD)
        self.register_buffer("in_scale", a.view(1, 3, 1, 1), persistent=False)
        self.register_buffer("in_shift", b.view(1, 3, 1, 1), persistent=False)
        if init_path is not None:
            self.load_clip_init(init_path)

    def load_clip_init(self, path):
        init = torch.load(path, map_location="cpu", weights_only=False)
        visual = dict(init["visual"])
        visual["positional_embedding"] = resize_pos_embed(visual["positional_embedding"],
                                                          self.visual.grid)
        self.visual.load_state_dict(visual)
        t_pos = F.normalize(init["text_pos"].float(), dim=-1)
        t_neg = F.normalize(init["text_neg"].float(), dim=-1)
        with torch.no_grad():
            self.head.weight.copy_(init.get("logit_scale", 100.0) * (t_pos - t_neg))
            self.head.bias.zero_()

    def forward(self, x):
        x = x * self.in_scale + self.in_shift
        f = F.normalize(self.visual(x).float(), dim=-1)
        return self.head(f)

    def param_groups(self, lr, head_lr, layer_decay, weight_decay):
        """AdamW groups with layer-wise LR decay; no weight decay on norms/biases/embeddings."""
        layers = len(self.visual.transformer.resblocks)
        groups = {}

        def depth(name):
            if name.startswith("visual.transformer.resblocks."):
                return int(name.split(".")[3]) + 1
            if name.startswith(("visual.conv1", "visual.class_embedding",
                                "visual.positional_embedding", "visual.ln_pre")):
                return 0
            return layers + 1  # ln_post, proj

        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("head."):
                key, g_lr = ("head", 0), head_lr
            else:
                d = depth(name)
                key, g_lr = ("layer", d), lr * layer_decay ** (layers + 1 - d)
            no_wd = p.ndim < 2 or "embedding" in name
            key = key + (no_wd,)
            if key not in groups:
                groups[key] = {"params": [], "lr": g_lr,
                               "weight_decay": 0.0 if no_wd else weight_decay}
            groups[key]["params"].append(p)
        return list(groups.values())
