"""Versioned, crash-safe backup of training artifacts.

Copies every new or changed artifact under runs/ (checkpoints, held-out
logits, logs, decoder.json) and the submission zips into backups/, keeping
the last --keep versions of each file:

    backups/runs/convnext_tiny_v1/PETA/model.pth/20261019-224512.pth

A file is backed up only if it is complete and readable: no *.tmp files,
untouched for --settle seconds (except append-only .log/.jsonl text), and .pth/.zip/.npy files must pass an
integrity check (zip CRCs / npy header). Each copy goes to a .tmp name, is
fsynced, then renamed, so a power cut during the backup can corrupt neither
the source nor an older backup.

Runs every 10 minutes from the systemd user timer installed by
install_backup_timer.sh; safe to run by hand at any time:

    python backup_artifacts.py            # one pass
    python backup_artifacts.py --list     # show what is backed up
"""
import argparse
import os
import shutil
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKUP_ROOT = ROOT / "backups"
PATTERNS = [
    "runs/**/*.pth", "runs/**/*.npy", "runs/**/*.json", "runs/**/*.jsonl",
    "runs/*.log", "submission_*.zip",
]


def is_intact(path):
    suffix = path.suffix
    try:
        if suffix in (".pth", ".zip"):  # torch checkpoints are zip archives
            with zipfile.ZipFile(path) as zf:
                return zf.testzip() is None
        if suffix == ".npy":
            with open(path, "rb") as f:
                return f.read(6) == b"\x93NUMPY"
        return True
    except (zipfile.BadZipFile, OSError):
        return False


def fsync_dir(d):
    fd = os.open(d, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def safe_copy(src, dst):
    tmp = dst.with_name(dst.name + ".tmp")
    with open(src, "rb") as fi, open(tmp, "wb") as fo:
        shutil.copyfileobj(fi, fo, 16 << 20)
        fo.flush()
        os.fsync(fo.fileno())
    shutil.copystat(src, tmp)
    os.replace(tmp, dst)
    fsync_dir(dst.parent)


def backup_once(keep, settle):
    now = time.time()
    copied, skipped = 0, []
    for pattern in PATTERNS:
        for src in sorted(ROOT.glob(pattern)):
            if not src.is_file() or src.name.endswith(".tmp"):
                continue
            st = src.stat()
            # text logs are appended to constantly; a copy mid-write is still useful
            if now - st.st_mtime < settle and src.suffix not in (".log", ".jsonl"):
                skipped.append(f"{src.relative_to(ROOT)} (still being written)")
                continue
            vdir = BACKUP_ROOT / src.relative_to(ROOT)
            vdir.mkdir(parents=True, exist_ok=True)
            versions = sorted(p for p in vdir.iterdir() if not p.name.endswith(".tmp"))
            if versions:
                last = versions[-1].stat()
                if last.st_size == st.st_size and int(last.st_mtime) == int(st.st_mtime):
                    continue  # unchanged since the latest backup
            if not is_intact(src):
                skipped.append(f"{src.relative_to(ROOT)} (failed integrity check)")
                continue
            stamp = datetime.fromtimestamp(st.st_mtime).strftime("%Y%m%d-%H%M%S")
            safe_copy(src, vdir / f"{stamp}{src.suffix}")
            copied += 1
            versions = sorted(p for p in vdir.iterdir() if not p.name.endswith(".tmp"))
            for old in versions[:-keep]:
                old.unlink()
    for p in BACKUP_ROOT.rglob("*.tmp"):  # leftovers from an interrupted pass
        p.unlink(missing_ok=True)
    # A finished fold (done.json) never resumes, so its last.pth backups are dead weight.
    for vdir in BACKUP_ROOT.glob("runs/*/*/last.pth"):
        fold = ROOT / vdir.parent.relative_to(BACKUP_ROOT)
        if (fold / "done.json").exists():
            shutil.rmtree(vdir)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{stamp}] backup: {copied} file(s) copied" +
          ("".join(f"\n  skipped {s}" for s in skipped) if skipped else ""))


def list_backups():
    for vdir in sorted({p.parent for p in BACKUP_ROOT.rglob("*") if p.is_file()}):
        files = sorted(vdir.iterdir())
        size = sum(f.stat().st_size for f in files) / 1e6
        print(f"{vdir.relative_to(BACKUP_ROOT)}: {len(files)} version(s), {size:.0f} MB, "
              f"latest {files[-1].name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", type=int, default=3, help="versions kept per file")
    ap.add_argument("--settle", type=float, default=30, help="skip files modified this recently (s)")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()
    if args.list:
        list_backups()
        return
    lock = BACKUP_ROOT / ".lock"
    BACKUP_ROOT.mkdir(exist_ok=True)
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        if time.time() - lock.stat().st_mtime < 3600:
            print("another backup pass is running; skipping")
            return
        lock.unlink()  # stale lock from a pass killed by a power cut
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    try:
        backup_once(args.keep, args.settle)
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
