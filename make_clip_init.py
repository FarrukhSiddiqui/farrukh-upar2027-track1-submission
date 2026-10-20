"""Convert OpenAI CLIP weights + attribute prompts into pretrained/<name>_init.pth.

Local, one-off tool (needs `pip install --no-deps open_clip_torch ftfy regex`
for the tokenizer and text tower). The output holds only what training and
the submission need: the visual tower's state dict and, per attribute, the
text embeddings of a positive and a negative prompt.

    python make_clip_init.py --name clip_vit_b16 --ckpt pretrained/ViT-B-16.pt
"""
import argparse

import torch

# (positive phrase, negative phrase) per attribute, in challenge order.
TEMPLATE = "a surveillance photo of a pedestrian {}."
UPPER = "wearing a {} shirt, sweater or jacket"
LOWER = "wearing {} trousers, shorts or skirt"
PROMPTS = [
    ("who is a child or teenager", "who is an adult"),
    ("who is an adult", "who is a child or an elderly person"),
    ("who is an elderly person", "who is a young person"),
    ("who is a woman", "who is a man"),
    ("with short hair", "with long hair"),
    ("with long hair", "with short hair"),
    ("who is bald", "with hair on the head"),
    ("wearing a short-sleeved top", "wearing a long-sleeved top"),
] + [(UPPER.format(c), UPPER.format("plain")) for c in
     ["black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white",
      "yellow", "multicoloured patterned"]] + [
    ("wearing shorts or a short skirt", "wearing long trousers"),
] + [(LOWER.format(c), LOWER.format("plain")) for c in
     ["black", "blue", "brown", "green", "grey", "orange", "pink", "purple", "red", "white",
      "yellow", "multicoloured patterned"]] + [
    ("wearing trousers or shorts", "wearing a skirt or a dress"),
    ("wearing a skirt or a dress", "wearing trousers or shorts"),
    ("carrying a backpack", "carrying nothing"),
    ("carrying a handbag or shoulder bag", "carrying nothing"),
    ("wearing glasses", "without glasses"),
    ("wearing sunglasses", "without glasses"),
    ("wearing a hat or cap", "with nothing on the head"),
]
assert len(PROMPTS) == 40


def main():
    import open_clip
    from open_clip.model import build_model_from_openai_state_dict

    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="clip_vit_b16")
    ap.add_argument("--ckpt", default="pretrained/ViT-B-16.pt")
    ap.add_argument("--arch", default="ViT-B-16")
    args = ap.parse_args()

    sd = torch.jit.load(args.ckpt, map_location="cpu").state_dict()
    model = build_model_from_openai_state_dict(sd).float().eval()
    tok = open_clip.get_tokenizer(args.arch)
    with torch.no_grad():
        pos = model.encode_text(tok([TEMPLATE.format(p) for p, _ in PROMPTS]))
        neg = model.encode_text(tok([TEMPLATE.format(n) for _, n in PROMPTS]))
    visual = {k[len("visual."):]: v.float() for k, v in sd.items() if k.startswith("visual.")}
    out = f"pretrained/{args.name}_init.pth"
    torch.save({"visual": visual, "text_pos": pos, "text_neg": neg,
                "logit_scale": float(sd["logit_scale"].exp()), "prompts": PROMPTS}, out)
    print(f"saved {out}: {len(visual)} visual tensors, text {tuple(pos.shape)}")


if __name__ == "__main__":
    main()
