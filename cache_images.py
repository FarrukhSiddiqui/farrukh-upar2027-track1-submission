"""Decode every train+val image once, resize to 256x128, store as a uint8 memmap.

Training then reads raw pixels and augments on the GPU, so JPEG decoding never
limits throughput. Resizing uses the same PIL call as submission/run.py.

    python cache_images.py
"""
from multiprocessing import Pool

import numpy as np
from PIL import Image
from tqdm import tqdm

from upar_common import CACHE_DIR, DATA_ROOT, DOMAINS, IMAGE_SIZE, domain_of, read_gt


def load_resized(rel_path):
    with Image.open(DATA_ROOT / rel_path) as im:
        im = im.convert("RGB").resize((IMAGE_SIZE[1], IMAGE_SIZE[0]), Image.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def main():
    paths, labels, names = [], [], None
    for split in ("train", "val"):
        p, y, names = read_gt(split)
        paths += p
        labels.append(y)
    labels = np.concatenate(labels)
    domains = np.array([DOMAINS.index(domain_of(p)) for p in paths], dtype=np.int64)

    CACHE_DIR.mkdir(exist_ok=True)
    out = np.memmap(CACHE_DIR / "images_u8.bin", dtype=np.uint8, mode="w+",
                    shape=(len(paths), IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
    with Pool() as pool:
        for i, arr in enumerate(tqdm(pool.imap(load_resized, paths, chunksize=256),
                                     total=len(paths), desc="Caching")):
            out[i] = arr
    out.flush()
    np.savez(CACHE_DIR / "meta.npz", labels=labels, domains=domains,
             names=np.array(names), paths=np.array(paths))
    for d, name in enumerate(DOMAINS):
        print(f"{name}: {(domains == d).sum()} images")


if __name__ == "__main__":
    main()
