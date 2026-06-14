#!/usr/bin/env python3
"""Download the public project dataset and model assets."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import tempfile
from urllib.request import urlopen


RELEASE_BASE = "https://github.com/DotNaos/fs26-crosswalk-detector/releases/download"

DATASET_ARCHIVE = "sam3-500k-static-metadata-v1.tar.gz"
DATASET_SHA256 = "1a51b3e25661acd322f6a183a82a33a5201b68da72a20c64f99e8cef4ba10993"
DATASET_URL = f"{RELEASE_BASE}/submission-dataset-v1/{DATASET_ARCHIVE}"

MODEL_RELEASE = f"{RELEASE_BASE}/crossmasknet-v4"
MODEL_FILES = {
    "crossmasknet_best.pt": "910fb5c07f85e4ab6816016efb9d120bc0753e7d4f3116e9db5778b4ec774e77",
    "metrics.json": "34dfa32ace4221dd28d8a7fd01b9a50c4c1d975cabda3185e67ee59b115cad4d",
    "road-filter-metrics.json": "771100e42ae33820d339d602c9221ab65e2841fddd867faa7de010ac076bf3f4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Crosswalk Detector release assets.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="Project root. Defaults to the current directory.")
    parser.add_argument(
        "--dataset-output",
        type=Path,
        default=None,
        help="Where to extract static dataset files. Defaults to web/public/static-datasets.",
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=None,
        help="Where to store CrossMaskNet v4 assets. Defaults to models/crossmask/sam3-500k-road-channel-v4.",
    )
    parser.add_argument("--download-dir", type=Path, default=None, help="Cache downloads here instead of a temporary directory.")
    parser.add_argument("--skip-dataset", action="store_true", help="Do not download the static dataset package.")
    parser.add_argument("--skip-model", action="store_true", help="Do not download the model checkpoint package.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset_output = args.dataset_output or repo_root / "web" / "public" / "static-datasets"
    model_output = args.model_output or repo_root / "models" / "crossmask" / "sam3-500k-road-channel-v4"

    if args.download_dir:
        download_dir = args.download_dir.resolve()
        download_dir.mkdir(parents=True, exist_ok=True)
        cleanup = None
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="crosswalk-assets-")
        download_dir = Path(cleanup.name)

    try:
        if not args.skip_dataset:
            download_dataset(download_dir, dataset_output, force=args.force)
        if not args.skip_model:
            download_model(download_dir, model_output, force=args.force)
    finally:
        if cleanup:
            cleanup.cleanup()

    print("Project assets are ready.")
    print(f"Dataset: {dataset_output}")
    print(f"Model:   {model_output}")
    return 0


def download_dataset(download_dir: Path, output: Path, *, force: bool) -> None:
    archive_path = download_dir / DATASET_ARCHIVE
    download_file(DATASET_URL, archive_path, expected_sha256=DATASET_SHA256, force=force)

    target_dataset = output / "sam3-500k-masks-v1"
    target_index = output / "index.json"
    if target_dataset.exists() and target_index.exists() and not force:
        print(f"Dataset already exists: {output}")
        return

    output.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r:gz") as archive:
        safe_extract(archive, output)
    print(f"Extracted dataset metadata to {output}")


def download_model(download_dir: Path, output: Path, *, force: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    for name, expected_hash in MODEL_FILES.items():
        target = output / name
        if target.exists() and not force:
            verify_sha256(target, expected_hash)
            print(f"Model asset already exists: {target}")
            continue
        download_file(f"{MODEL_RELEASE}/{name}", target, expected_sha256=expected_hash, force=True)

    sums = "\n".join(f"{value}  {name}" for name, value in MODEL_FILES.items()) + "\n"
    (output / "SHA256SUMS").write_text(sums, encoding="utf-8")


def download_file(url: str, target: Path, *, expected_sha256: str, force: bool) -> None:
    if target.exists() and not force:
        verify_sha256(target, expected_sha256)
        print(f"Download already exists: {target}")
        return

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".download")
    print(f"Downloading {url}")
    with urlopen(url) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    verify_sha256(temporary, expected_sha256)
    temporary.replace(target)
    print(f"Saved {target}")


def verify_sha256(path: Path, expected: str) -> None:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        if root != target and root not in target.parents:
            raise RuntimeError(f"Archive member escapes destination: {member.name}")
    archive.extractall(destination)


if __name__ == "__main__":
    raise SystemExit(main())
