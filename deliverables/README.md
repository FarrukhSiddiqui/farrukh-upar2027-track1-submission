# UPAR 2027 Track 1 — deliverables

Pedestrian attribute recognition under domain shift (RWS workshop, WACV 2027,
[Codabench competition 18180](https://www.codabench.org/competitions/18180/)).

**Final submission:** CLIP ViT-L/14 fine-tuned with leave-one-dataset-out
validation, WiSE-FT (α = 0.9) and a metric-aware decoder.
**Official result: Challenge_Avg 0.7856** (mA 0.788, instance F1 0.784),
rank 30 at submission time.

## Contents

| Path | What it is |
| --- | --- |
| `README.md` | This file: task, approach, results, how to reproduce |
| `submission_proof/final_clip_vitl14_wise0.9/` | Official scoring output of the final submission (`scores.json`, `per_attribute.json`, `detailed_results.html`, original zip) |
| `submission_proof/earlier_clip_vitb16_wise0.8/` | Same for the earlier CLIP ViT-B/16 submission (0.7663) |
| `results/model_results.csv` | Every model: LODO estimate, leaderboard scores, per test dataset, inference time |
| `results/ablations_lodo.csv` | Decoder, TTA, stacker, fusion and ensemble experiments (LODO) |
| `results/test_per_attribute.csv` | Per-attribute test mA and F1, final vs earlier model |
| `results/decoder_*.json` | Tuned decoder of each model, including its nested-LODO scores |
| `results/training_logs/` | Per-epoch held-out scores of every training run and fold |
| `code_upar_track1.zip` | Source code (no data, no weights) |

The report and the demo slides are separate documents.

## Task

Predict 40 binary attributes (age, gender, hair, clothing length, colour and
type, accessories) for each pedestrian crop. Training data: PA-100K, PETA and
Market1501 (131,076 labelled crops). The hidden test set comes from unseen
domains: MEVID (28,095 surveillance crops) and Synth (1,153 synthetic crops),
scored separately and averaged. The ranking metric is the harmonic mean of
label mA and instance F1, with an attribute present above probability 0.5.
Submissions are code, run on an A10 GPU with torch 2.4.1, no network and a
5-minute limit.

## Approach

1. **Leave-one-dataset-out (LODO) validation.** Each model is trained three
   times, each time holding out one source dataset, then once on all data.
   The held-out predictions drive every choice; the leaderboard is never used
   for tuning. The LODO estimate ranked all four submissions correctly, about
   0.05 above the test score.
2. **CLIP backbone in plain PyTorch** (`submission/clip_vit.py`), loaded from
   OpenAI weights, with a text-initialised attribute head (positive minus
   negative prompt embedding per attribute).
3. **Fine-tuning:** 1 head-only epoch + 4 full epochs, AdamW, layer-wise LR
   decay 0.8, drop-path 0.1, EMA, bf16; dataset-balanced sampling; loss masked
   on attributes a source never labels; GPU augmentation with low-resolution,
   motion-blur and JPEG simulation and no hue changes.
4. **WiSE-FT:** shipped weights = 0.9 × fine-tuned + 0.1 × zero-shot CLIP.
5. **Metric-aware set decoder:** per image, choose the attribute set that
   maximises expected mA gain + instance precision/recall; tuned on LODO
   predictions and baked into the output probabilities.

## Results

| Model | LODO estimate | Leaderboard Avg |
| --- | --- | --- |
| **CLIP ViT-L/14 + WiSE-FT 0.9 (final)** | 0.829 | **0.7856** |
| CLIP ViT-B/16 + WiSE-FT 0.8 | 0.816 | 0.7663 |
| CLIP ViT-B/16 | 0.812 | 0.7605 |
| ConvNeXt-Tiny (ImageNet) | 0.787 | 0.7347 |
| CLIP ViT-L/14 + SigLIP2 ViT-L/16 ensemble | 0.834 | not uploaded (Codabench down) |

## How to reproduce

Environment: Linux, AMD GPU with ROCm 7.1 (any CUDA GPU works too),
Python 3.14 with torch 2.13 and torchvision. `open_clip_torch` is needed only
to convert the CLIP weights (`pip install --no-deps open_clip_torch ftfy regex`).

```bash
python download_datasets.py            # official starter-kit download
python cache_images.py                 # decode + resize all crops once (12.9 GB memmap)
python make_clip_init.py --name clip_vit_l14 --ckpt pretrained/ViT-L-14.pt --arch ViT-L-14
./run_pipeline.sh clip_l14_v1 --backbone clip_vit_l14 --epochs 5 --batch-size 32 \
  --lr 2e-5 --head-lr-mult 50 --layer-decay 0.8 --lp-epochs 1 --drop-path 0.1 \
  --weight-decay 0.05 --clip-grad 1.0 --ema 0.999 --mask-unlabeled --strong-lowres --jpeg-p 0.3
python wise_ft.py --run clip_l14_v1 --alphas 0.9 --no-tta --suffix _nt
python fit_decoder.py --run clip_l14_v1_wise0.9_nt
python make_submission.py --run clip_l14_v1_wise0.9_nt   # -> submission_clip_l14_v1_wise0.9_nt.zip
```

`pretrained/ViT-L-14.pt` is OpenAI's CLIP checkpoint
(`openaipublic.azureedge.net/clip/models/b8cca3fd.../ViT-L-14.pt`, SHA-256
`b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836`).
A full ViT-L run (3 folds + final model) takes about 4.5 hours on an
RX 9070 XT. Re-running the same command after a crash resumes from the last
epoch. `python emulate_ingestion.py --submission <unzipped dir> --work <tmp>`
runs a zip the way the Codabench ingestion program does; run it in a
Python 3.11 / torch 2.4.1 environment to match the evaluation container.
