#!/usr/bin/env bash
# Download OpenAI's CLIP ViT-L/14 checkpoint, check its SHA-256, and build
# pretrained/clip_vit_l14_init.pth (visual weights + text-initialised head).
# Needs open_clip only for this one step (see requirements-train.txt).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
mkdir -p pretrained
URL=https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt
SHA=b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836
[ -f pretrained/ViT-L-14.pt ] || curl -L --fail -o pretrained/ViT-L-14.pt "$URL"
echo "$SHA  pretrained/ViT-L-14.pt" | sha256sum -c -
$PY make_clip_init.py --name clip_vit_l14 --ckpt pretrained/ViT-L-14.pt --arch ViT-L-14
