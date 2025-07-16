#!/usr/bin/env python3
"""
Download Wikipedia pages listed in a CSV file and store them
in a local corpus, prefixing each filename with the row id.

Example
-------
$ python download_wiki.py nq_sc.csv --output corpus --workers 16
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import csv
import os
import sys
from pathlib import Path
from typing import Iterable, Mapping, Sequence
from urllib.parse import unquote, urlparse

import requests

DEFAULT_LANG_COLS: Mapping[str, str] = {
    "en": "url_en",
    "it": "url_it",
    "sc": "url_sc",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Download Wikipedia pages from CSV.")
    parser.add_argument("csv_file", help="Path to CSV file.")
    parser.add_argument(
        "-o",
        "--output",
        default="corpus",
        help="Output directory with language sub-dirs (default: corpus).",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=os.cpu_count() * 5,
        help="Parallel download workers (default: 5×CPU cores).",
    )
    parser.add_argument(
        "--lang-cols",
        default="en:url_en,it:url_it,sc:url_sc",
        help="Comma-separated mapping language:column.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download and overwrite existing files.",
    )
    return parser.parse_args(argv)


def read_rows(csv_path: Path) -> Iterable[dict[str, str]]:
    """Yield CSV rows as dicts."""
    with csv_path.open(newline="", encoding="utf-8") as fh:
        yield from csv.DictReader(fh)


def page_title(url: str) -> str:
    """Return Wikipedia page title extracted from *url*."""
    path = urlparse(url).path          # '/wiki/Dubai'
    return unquote(path.rsplit("/", 1)[-1]) or "index"


def download(url: str, dest: Path, overwrite: bool) -> None:
    """Download *url* into *dest*."""
    if dest.exists() and not overwrite:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, timeout=20)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
    except requests.RequestException as exc:
        print(f"Error downloading {url!r}: {exc}", file=sys.stderr)


def build_jobs(
    rows: Iterable[dict[str, str]],
    lang_cols: Mapping[str, str],
    out_dir: Path,
    overwrite: bool,
) -> list[tuple[str, Path]]:
    """Create (url, dest) tuples for all downloads."""
    jobs: list[tuple[str, Path]] = []
    for row in rows:
        prefix = row.get("id", "").strip()
        if not prefix:
            continue
        for lang, col in lang_cols.items():
            url = (row.get(col) or "").strip()
            if not url:
                continue
            filename = f"{prefix}_{page_title(url)}.html"
            dest = out_dir / lang / filename
            jobs.append((url, dest))
    return jobs


def main(argv: Sequence[str] | None = None) -> None:
    """Run script."""
    args = parse_args(argv)
    lang_cols = dict(pair.split(":", 1) for pair in args.lang_cols.split(",") if ":" in pair)
    rows = list(read_rows(Path(args.csv_file)))
    jobs = build_jobs(rows, lang_cols, Path(args.output), args.overwrite)

    with futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        pool.map(lambda job: download(*job, overwrite=args.overwrite), jobs)


if __name__ == "__main__":
    main()