from __future__ import annotations

import argparse
import json
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = ROOT / ".cache" / "wordle-lexicon"

SOURCES = {
    "kaikki": {
        "url": "https://kaikki.org/dictionary/English/kaikki.org-dictionary-English.jsonl.gz",
        "filename": "kaikki.org-dictionary-English.jsonl.gz",
    },
    "moby_pos_ii": {
        "url": "https://www.gutenberg.org/files/3203/files.zip",
        "filename": "moby-pos-ii.zip",
    },
    "gcide": {
        "url": "https://ftp.gnu.org/gnu/gcide/gcide-0.54.tar.xz",
        "filename": "gcide-0.54.tar.xz",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def download(
    url: str,
    destination: Path,
    *,
    force: bool,
    previous: dict[str, str | int] | None,
) -> tuple[dict[str, str | int], str]:
    if destination.exists() and not force:
        metadata = previous or {
            "url": url,
            "filename": destination.name,
            "size_bytes": destination.stat().st_size,
        }
        metadata.pop("download_status", None)
        return metadata, "cached"

    temporary = destination.with_suffix(f"{destination.suffix}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "tiny-wordle-lab-v2"})
    with urllib.request.urlopen(request, timeout=180) as response:
        with temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        metadata = {
            "url": response.geturl(),
            "filename": destination.name,
            "size_bytes": temporary.stat().st_size,
        }
        for header, key in (
            ("ETag", "etag"),
            ("Last-Modified", "last_modified"),
        ):
            if value := response.headers.get(header):
                metadata[key] = value
    temporary.replace(destination)
    return metadata, "downloaded"


def extract_moby(archive: Path, cache_dir: Path) -> None:
    destination = cache_dir / "mobypos.txt"
    with zipfile.ZipFile(archive) as source:
        with source.open("mobypos.txt") as input_stream:
            with destination.open("wb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream)


def extract_gcide(archive: Path, cache_dir: Path) -> None:
    destination = cache_dir / "gcide-0.54"
    if destination.exists():
        shutil.rmtree(destination)
    with tarfile.open(archive) as source:
        members = [
            member
            for member in source.getmembers()
            if member.name.startswith("gcide-0.54/CIDE.")
        ]
        source.extractall(cache_dir, members=members, filter="data")


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = args.cache_dir / "sources.json"
    previous_metadata = (
        json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    )
    metadata = {}
    statuses = {}
    for name, source in SOURCES.items():
        destination = args.cache_dir / source["filename"]
        metadata[name], statuses[name] = download(
            source["url"],
            destination,
            force=args.force,
            previous=previous_metadata.get(name),
        )

    extract_moby(args.cache_dir / SOURCES["moby_pos_ii"]["filename"], args.cache_dir)
    extract_gcide(args.cache_dir / SOURCES["gcide"]["filename"], args.cache_dir)
    metadata_path.write_text(f"{json.dumps(metadata, indent=2)}\n")

    for name, details in metadata.items():
        print(f"{name}: {statuses[name]} ({details['size_bytes']:,} bytes)")


if __name__ == "__main__":
    main()
