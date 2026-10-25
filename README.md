# UPAR 2027 Challenge — Track 1: Pedestrian Attribute Recognition

Code for our Track 1 entry (RWS workshop @ WACV 2027,
[Codabench 18180](https://www.codabench.org/competitions/18180/)):
**CLIP ViT-L/14 fine-tuned with leave-one-dataset-out (LODO) model selection,
WiSE-FT and a metric-aware set decoder.**

| | Challenge_Avg | mA | Label F1 | Inst Acc | Inst Prec | Inst Rec | Inst F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Final submission 1007093 (`farrukh-siddiqui`) | **0.7856** | 0.7877 | 0.5504 | 0.6723 | 0.7531 | 0.8343 | 0.7835 |

Team: Farrukh Ahmed Siddiqui, Syed Muhammad Arham Tahir, Syed Muhammad Hamza Naveed
(Department of Artificial Intelligence, FAST-NUCES City Campus, Karachi).
The method is described in the fact sheet (`factsheet/`).

## Trained weights

Download from Google Drive: **[submission_clip_l14_v1_wise0.9_nt.zip](https://drive.google.com/file/d/1lKq8vwCnrUSBKOp9rm5ti3g1gAN0YDBd/view?usp=sharing)**
(537 MB, SHA-256 `e82373516fce2a3c8c21a49d02193ee9cbc5e925d5c8bc1b76747e961f27758a`).

This is **the exact zip scored on Codabench**: `run.py`, the model code, the
fp16 weights of the final model and the tuned decoder. From the command line:
`pip install gdown && gdown 1lKq8vwCnrUSBKOp9rm5ti3g1gAN0YDBd`.

The training start point (`pretrained/clip_vit_l14_init.pth`) is rebuilt from
OpenAI's public CLIP weights by `get_clip_weights.sh`. The fp32 checkpoints
(fine-tuned model before WiSE-FT, 1.2 GB each) are available on request.

## Reproduce the leaderboard predictions (inference)

The submission zip is self-contained and needs only torch, torchvision,
numpy and Pillow. With Docker (same base image as the evaluation container):

```bash
docker build -t upar-t1 .
unzip submission_clip_l14_v1_wise0.9_nt.zip -d weights/submission
docker run --rm --gpus all -v "$PWD/weights:/weights" -v /path/to/input_data:/input \
  upar-t1 python predict.py --submission /weights/submission \
  --images /input/images --list /input/images.csv --out /weights/predictions.csv
```

`--list` takes the Codabench `images.csv` layout (one path per line, relative to
`--images`); without it every image under `--images` is used. The output has the
`gt.csv` layout with 40 probabilities per image (present when > 0.5). The zip
itself is also the Codabench submission format: the ingestion program imports
`run.py` and calls `load_model()` / `predict_batch()`. The 29,248 test images
took 101.5 s on the A10.

Without Docker: Python ≥ 3.11 with torch ≥ 2.4 and torchvision, then the same
`python predict.py …` command.

The image (about 9.5 GB, torch 2.4.1 from the base image left untouched) was
built and tested with Docker 29: inference with the released zip, a short
training run, and the CLIP weight conversion all run inside the container.

## Reproduce training

Hardware used: one AMD RX 9070 XT (16 GB, ROCm 7.1, PyTorch 2.13). The code is
device-agnostic and runs on CUDA as is (bf16 autocast). A full run (3 LODO folds
+ final model) took about 4.5 hours; it needs about 13 GB of GPU memory at batch 32
and about 15 GB of disk for the image cache.

```bash
docker run --rm -it --gpus all --shm-size 8g -v "$PWD:/workspace" upar-t1 bash
# inside the container, in /workspace:
python download_datasets.py              # official starter-kit data -> data/
python cache_images.py                   # decode + resize all 131,076 crops once -> cache/
./get_clip_weights.sh                    # OpenAI CLIP ViT-L/14 -> pretrained/clip_vit_l14_init.pth
./run_pipeline.sh clip_l14_v1 --backbone clip_vit_l14 --epochs 5 --batch-size 32 \
  --lr 2e-5 --head-lr-mult 50 --layer-decay 0.8 --lp-epochs 1 --drop-path 0.1 \
  --weight-decay 0.05 --clip-grad 1.0 --ema 0.999 --mask-unlabeled --strong-lowres --jpeg-p 0.3
python wise_ft.py --run clip_l14_v1 --alphas 0.9 --no-tta --suffix _nt
python fit_decoder.py --run clip_l14_v1_wise0.9_nt
python make_submission.py --run clip_l14_v1_wise0.9_nt    # -> submission_clip_l14_v1_wise0.9_nt.zip
```

What each step does:

1. `run_pipeline.sh` trains three LODO folds (each holds out one of Market1501,
   PA-100K, PETA), then the final model on all data (`runs/clip_l14_v1/none/`),
   and writes each fold's held-out logits. It resumes from the last epoch if
   interrupted.
2. `wise_ft.py` interpolates every model 90/10 with zero-shot CLIP and
   re-predicts the held-out folds without flip TTA.
3. `fit_decoder.py` tunes the set decoder on those held-out logits and reports
   the nested-LODO estimate (0.829 for the submitted model).
4. `make_submission.py` packages `submission/` with the fp16 weights and the
   decoder into the Codabench zip.

Seed 42 is fixed; GPU kernels are not bit-deterministic, so retrained weights
differ slightly from the released ones.

## Repository layout

| Path | Purpose |
| --- | --- |
| `submission/run.py` | Codabench entry point: loading, batched fp16 inference, set decoder |
| `submission/clip_vit.py` | CLIP ViT in plain PyTorch + text-initialised attribute head |
| `train_lodo.py` | LODO training: GPU augmentation, sampling, masked BCE, EMA, resumable checkpoints |
| `fit_decoder.py` | Per-attribute thresholds and the per-image set decoder, nested-LODO evaluation |
| `wise_ft.py`, `soup.py`, `ensemble_eval.py` | WiSE-FT, weight soups, weighted logit ensembles |
| `make_clip_init.py`, `get_clip_weights.sh` | Convert OpenAI CLIP weights and build the prompt-initialised head |
| `cache_images.py`, `upar_common.py` | Image cache, model factory, metrics, atomic I/O |
| `predict.py`, `emulate_ingestion.py`, `test_submission.py` | Run a submission locally the way Codabench does |
| `submission/siglip_vit.py`, `make_siglip_init.py` | SigLIP2 ViT-L/16 backbone (experiments, not in the final entry) |
| `factsheet/` | Challenge fact sheet (LaTeX) |
| `docs/STARTER_KIT_README.md` | The organizers' original starter-kit README |

## Data and licences

Uses only the organizers' training data (PA-100K, PETA, Market1501 with UPAR
annotations; CC BY-NC-SA 3.0) and OpenAI's public CLIP weights. Please cite the
UPAR papers listed in `docs/STARTER_KIT_README.md`.
