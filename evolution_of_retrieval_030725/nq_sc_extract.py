#!/usr/bin/env python3
"""
Stream KILT Natural Questions and keep only questions whose English Wikipedia
page has a Sardinian (“sc”) version.

Interrupt-safe **and** parallel (thread pool).  
Duplicates are recognised by the dataset *id*.

CSV columns
-----------
dataset, id, question, short_answer, long_answer, url_en, url_it, url_sc
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import csv
import time
import urllib.parse as ulib
from pathlib import Path
from typing import Iterable

import requests
from datasets import load_dataset
from tqdm import tqdm

API = "https://{lang}.wikipedia.org/w/api.php"
HEADERS = [
    "dataset",
    "id",
    "question",
    "short_answer",
    "long_answer",
    "url_en",
    "url_it",
    "url_sc",
]


# ──────────────────────────── Wikipedia helpers ────────────────────────────
def get_langlinks(title: str, src: str = "en") -> dict[str, str]:
    """Return a *lang → title* mapping of inter-language links for *title*."""
    r = requests.get(
        API.format(lang=src),
        params={
            "action": "query",
            "titles": title,
            "prop": "langlinks",
            "lllimit": 500,
            "format": "json",
        },
        timeout=10,
    )
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return {ll["lang"]: ll["*"] for ll in page.get("langlinks", [])}


def wiki_url(title: str, lang: str) -> str:
    """Return full HTTPS URL for a Wikipedia page."""
    escaped = ulib.quote(title.replace(" ", "_"))
    return f"https://{lang}.wikipedia.org/wiki/{escaped}"


def fetch_plaintext(title: str, lang: str = "en") -> str:
    """Download plaintext extract of *title*."""
    r = requests.get(
        API.format(lang=lang),
        params={
            "action": "query",
            "prop": "extracts",
            "explaintext": 1,
            "titles": title,
            "format": "json",
        },
        timeout=10,
    )
    r.raise_for_status()
    page = next(iter(r.json()["query"]["pages"].values()))
    return page.get("extract", "")


def pick_long_answer(article: str, short: str) -> str:
    """
    Return first paragraph that contains *short* (case-insensitive);
    fallback to the first paragraph.
    """
    paragraphs: Iterable[str] = (p.strip() for p in article.split("\n\n") if p.strip())
    for p in paragraphs:
        if short and short.lower() in p.lower():
            return p
    return next(iter(paragraphs), "")


def canonical_short(ans_raw) -> str:
    """Normalise raw short answer from a KILT-NQ sample."""
    if isinstance(ans_raw, str):
        return ans_raw.strip()
    if isinstance(ans_raw, list):
        if len(ans_raw) == 1:
            return ans_raw[0].strip()
        if all(len(tok) == 1 for tok in ans_raw):
            return "".join(ans_raw).strip()
        return " ".join(tok.strip() for tok in ans_raw).strip()
    return ""


# ───────────────────────── stateful resume helpers ─────────────────────────
def load_done(path: Path) -> set[str]:
    """
    Return IDs already present in *path*.
    If the CSV lacks the `id` column (legacy file), fall back to `question`.
    """
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as fh:
        rdr = csv.DictReader(fh)
        key = "id" if "id" in (rdr.fieldnames or []) else "question"
        return {row[key] for row in rdr}


# ───────────────────────────── worker function ─────────────────────────────
def build_row(ex: dict, pause: float) -> list[str] | None:
    """Return a CSV row or None if the example must be skipped."""
    q_id: str = ex["id"]
    question: str = ex["input"]

    title_en: str | None = None
    short_answer: str = ""

    # First provenance + answer
    for out in ex.get("output", []):
        if out.get("provenance"):
            title_en = out["provenance"][0].get("title")
            short_answer = canonical_short(out.get("answer", ""))
            break
    if not title_en:  # missing provenance
        return None

    # Language links (needs sc)
    try:
        ll = get_langlinks(title_en)
    except requests.RequestException:
        return None
    if "sc" not in ll:
        return None

    # Long answer (best effort)
    try:
        article = fetch_plaintext(title_en)
        long_answer = pick_long_answer(article, short_answer)
    except requests.RequestException:
        long_answer = ""

    time.sleep(pause)  # soft-rate-limit

    return [
        "natural_questions",
        q_id,
        question,
        short_answer,
        long_answer,
        wiki_url(title_en, "en"),
        wiki_url(ll["it"], "it") if "it" in ll else "",
        wiki_url(ll["sc"], "sc"),
    ]


# ─────────────────────────────────── main ───────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile", required=True, help="Output CSV path")
    parser.add_argument("--max", type=int, help="Max rows to *keep* this run")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent Wikipedia fetchers (threads)",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.05,
        help="Sleep inside each worker after API calls (s)",
    )
    args = parser.parse_args()
    out_path = Path(args.outfile)

    done_ids = load_done(out_path)

    ds = load_dataset("facebook/kilt_tasks", name="nq", split="train", streaming=True)

    mode = "a" if out_path.exists() else "w"
    with out_path.open(mode, newline="", encoding="utf-8") as fh, tqdm(
        unit="q", initial=len(done_ids)
    ) as bar, cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        wr = csv.writer(fh, lineterminator="\n")
        if mode == "w":
            wr.writerow(HEADERS)

        kept = 0
        futures: list[cf.Future] = []

        def flush(fs: list[cf.Future]) -> list[cf.Future]:
            """Write completed futures, return the still-pending ones."""
            nonlocal kept
            pending: list[cf.Future] = []
            for fut in fs:
                if fut.done():
                    bar.update(1)
                    row = fut.result()
                    if row:
                        wr.writerow(row)
                        fh.flush()
                        kept += 1
                else:
                    pending.append(fut)
            return pending

        for ex in ds:
            if args.max and kept >= args.max:
                break
            if ex["id"] in done_ids:
                bar.update(1)
                continue
            futures.append(pool.submit(build_row, ex, args.pause))
            if len(futures) >= args.workers * 4:
                futures = flush(futures)

        # Drain remaining tasks
        while futures:
            futures = flush(futures)
            time.sleep(0.01)

    print(f"✅ Added {kept} new rows to {out_path}")


if __name__ == "__main__":
    main()