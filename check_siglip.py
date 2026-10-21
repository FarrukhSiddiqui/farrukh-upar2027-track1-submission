"""CPU checks for submission/siglip_vit.py against open_clip's SigLIP2.

1. identical image embeddings at the native 256x256 (same weights, random input);
2. zero-shot mA of the text-initialised head on a sample of training images
   at our 256x128 input (sanity check of prompts / tokenizer / pos-embed resize).

    HIP_VISIBLE_DEVICES= python check_siglip.py
"""
import sys

import numpy as np
import torch

sys.path.insert(0, "submission")
from siglip_vit import (IMAGENET_MEAN, IMAGENET_STD, SIGLIP_MEAN, SIGLIP_STD,  # noqa: E402
                        SigLIPAttributeNet)
from upar_common import load_cache, per_domain_metrics  # noqa: E402

torch.set_num_threads(8)


def main():
    import open_clip
    ref = open_clip.create_model("ViT-L-16-SigLIP2-256",
                                 pretrained="pretrained/siglip2_l16_256/open_clip_model.safetensors")
    ref = ref.float().eval()
    mine = SigLIPAttributeNet("siglip2_l16", image_size=(256, 256),
                              init_path="pretrained/siglip2_l16_init.pth").eval()
    x = torch.randn(2, 3, 256, 256)  # SigLIP-normalised input for the reference
    m, s = torch.tensor(SIGLIP_MEAN).view(1, 3, 1, 1), torch.tensor(SIGLIP_STD).view(1, 3, 1, 1)
    im, istd = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        a = ref.encode_image(x)
        b = mine.visual((x * s + m - im) / istd * mine.in_scale + mine.in_shift)
    print(f"max |diff| vs open_clip: {(a - b).abs().max().item():.2e} (feature scale {a.abs().mean().item():.3f})")

    images, labels, domains, _, _ = load_cache()
    idx = np.concatenate([np.where(domains == k)[0][:400] for k in range(3)])
    model = SigLIPAttributeNet("siglip2_l16", image_size=(256, 128),
                               init_path="pretrained/siglip2_l16_init.pth").eval()
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    out = []
    with torch.no_grad():
        for i in range(0, len(idx), 50):
            u8 = torch.from_numpy(np.ascontiguousarray(images[idx[i:i + 50]]))
            out.append(model(((u8.permute(0, 3, 1, 2).float() / 255) - mean) / std))
    lg = torch.cat(out).numpy()
    # zero-shot: one global bias per attribute set so the predicted rate matches the prior
    pri = labels[idx].mean(0)
    thr = np.array([np.quantile(lg[:, a], 1 - pri[a]) for a in range(40)])
    res, _ = per_domain_metrics(labels[idx], lg > thr, domains[idx])
    print(f"zero-shot mA on {len(idx)} images (prior-matched cut): {res['mA']:.4f}  "
          f"[same check gives a rough sense of the text init; CLIP ViT-L at 0-cut was 0.654]")


if __name__ == "__main__":
    main()
