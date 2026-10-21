"""Convert open_clip SigLIP2 weights + attribute prompts into pretrained/<name>_init.pth.

Local, one-off tool (needs open_clip, timm, transformers for the reference
model and tokenizer). Keeps only what training and the submission need: the
vision trunk's state dict (timm names) and the text embeddings of a positive
and a negative prompt per attribute, as for make_clip_init.py.

    python make_siglip_init.py
"""
import argparse

import torch

from make_clip_init import PROMPTS, TEMPLATE


def main():
    import open_clip
    from open_clip.tokenizer import HFTokenizer

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="siglip2_l16")
    ap.add_argument("--arch", default="ViT-L-16-SigLIP2-256")
    ap.add_argument("--dir", default="pretrained/siglip2_l16_256")
    args = ap.parse_args()

    model = open_clip.create_model(args.arch, pretrained=f"{args.dir}/open_clip_model.safetensors")
    model = model.float().eval()
    cfg = open_clip.get_model_config(args.arch)["text_cfg"]
    tok = HFTokenizer(args.dir, context_length=cfg["context_length"],
                      **cfg.get("tokenizer_kwargs", {}))
    with torch.no_grad():
        pos = model.encode_text(tok([TEMPLATE.format(p) for p, _ in PROMPTS]))
        neg = model.encode_text(tok([TEMPLATE.format(n) for _, n in PROMPTS]))
    trunk = {k: v.float().clone() for k, v in model.visual.trunk.state_dict().items()}
    extra = [n for n, m in model.visual.named_children() if n != "trunk"
             and sum(p.numel() for p in m.parameters())]
    assert not extra, f"unexpected visual modules with weights: {extra}"
    out = f"pretrained/{args.name}_init.pth"
    torch.save({"visual": trunk, "text_pos": pos, "text_neg": neg,
                "logit_scale": float(model.logit_scale.exp()), "prompts": PROMPTS}, out)
    print(f"saved {out}: {len(trunk)} visual tensors, text {tuple(pos.shape)}, "
          f"logit_scale {float(model.logit_scale.exp()):.1f}")


if __name__ == "__main__":
    main()
