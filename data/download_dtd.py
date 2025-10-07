"""
Download and extract the Describable Textures Dataset (DTD).
Source: https://www.robots.ox.ac.uk/~vgg/data/dtd/
47 texture categories, 5,640 images total (120 per category).
"""
import argparse
import os
import tarfile
import urllib.request
from pathlib import Path

DTD_URL = "https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz"
DTD_ARCHIVE = "dtd-r1.0.1.tar.gz"


def _progress_hook(count, block_size, total_size):
    downloaded = count * block_size
    pct = min(100.0, downloaded * 100.0 / total_size)
    print(f"\r  downloading: {pct:.1f}%  ({downloaded // 1024 // 1024} MB)", end="", flush=True)


def download_dtd(dest_dir: str = "data") -> Path:
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    archive_path = dest / DTD_ARCHIVE
    dtd_dir = dest / "dtd"

    if dtd_dir.exists():
        print(f"DTD already extracted at {dtd_dir}")
        return dtd_dir

    if not archive_path.exists():
        print(f"Downloading DTD from {DTD_URL} ...")
        urllib.request.urlretrieve(DTD_URL, archive_path, reporthook=_progress_hook)
        print()

    print(f"Extracting {archive_path} ...")
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(path=dest)

    print(f"DTD extracted to {dtd_dir}")
    archive_path.unlink()  # remove archive to save space
    return dtd_dir


def verify_dtd(dtd_dir: Path) -> bool:
    images_dir = dtd_dir / "images"
    if not images_dir.exists():
        return False
    categories = list(images_dir.iterdir())
    print(f"Found {len(categories)} categories in DTD.")
    total = sum(len(list(c.glob("*.jpg"))) for c in categories if c.is_dir())
    print(f"Total images: {total}")
    return len(categories) == 47


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download the DTD dataset")
    parser.add_argument("--dest", default="data", help="Destination directory")
    args = parser.parse_args()

    dtd_path = download_dtd(args.dest)
    ok = verify_dtd(dtd_path)
    if ok:
        print("DTD download and verification complete.")
    else:
        print("WARNING: DTD verification failed — unexpected directory structure.")
