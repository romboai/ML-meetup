#!/usr/bin/env python
"""Bulk paragraph extractor for a local Wikipedia HTML corpus.

Record example
--------------
{
    "paragraph_id": "-39873198756871997_1",
    "lang": "it",
    "page_title": "Cuore",
    "text": "Cuore è un romanzo pubblicato nel 1886..."
}

python extract_para.py \
    --root corpus \
    --langs it en sc \
    -o paragraphs.jsonl \
    --jobs 8
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import logging
import re
from pathlib import Path
from typing import Iterable, Sequence

import orjson
from bs4 import BeautifulSoup
from tqdm import tqdm


# --------------------------------------------------------------------------- #
#                           HTML  ➜  PARAGRAPHS                               #
# --------------------------------------------------------------------------- #
def _clean_soup(soup: BeautifulSoup) -> None:
    """Remove boilerplate elements in-place (references, navboxes, etc.)."""
    for sup in soup.select("sup.reference"):
        sup.decompose()
    for span in soup.select("span.mw-editsection"):
        span.decompose()
    for toc in soup.select("div#toc"):
        toc.decompose()

    selectors = (
        ".infobox", ".navbox", ".vertical-navbox", ".sidebar",
        ".metadata", ".ambox", ".hatnote",
        ".succession-box", "table.succession-box",
        ".plainlist.hlist", "table.plainlinks",
    )
    for sel in selectors:
        for el in soup.select(sel):
            el.decompose()


def _strip_after_references(content_div) -> None:
    for hdr in content_div.find_all(["h2", "h3"]):
        txt = hdr.get_text(strip=True).lower()
        if txt.startswith(("references", "note", "notes", "bibliografia")):
            for sib in list(hdr.next_siblings):
                sib.decompose()
            hdr.decompose()
            break


def paragraphs_from_html(html_path: Path) -> list[str]:
    """Return a list of clean paragraphs from a Wikipedia HTML page."""
    soup = BeautifulSoup(html_path.read_bytes(), "lxml")

    content = soup.select_one("div#mw-content-text")
    if content is None:
        raise ValueError("mw-content-text not found in HTML page")

    _clean_soup(content)
    _strip_after_references(content)

    def _is_valid(text: str) -> bool:
        return len(text) > 40 and text.count("·") < 3

    raw_paras: Iterable[str] = (p.get_text(" ", strip=True) for p in content.find_all("p"))
    return [re.sub(r"\s{2,}", " ", p) for p in raw_paras if _is_valid(p)]


# --------------------------------------------------------------------------- #
#                            FILENAME HELPERS                                 #
# --------------------------------------------------------------------------- #
def _parse_filename(path: Path) -> tuple[str, str]:
    """Return (doc_id, page_title) from a Wikipedia HTML filename."""
    doc_id, raw_title = path.stem.split("_", 1)
    return doc_id, raw_title.replace("_", " ")


# --------------------------------------------------------------------------- #
#                          PARALLEL FILE PROCESSING                           #
# --------------------------------------------------------------------------- #
def _process_file(path: Path) -> list[dict]:
    lang = path.parent.name
    doc_id, page_title = _parse_filename(path)
    paras = paragraphs_from_html(path)

    return [
        {
            "paragraph_id": f"{doc_id}_{idx}",
            "lang": lang,
            "page_title": page_title,
            "text": para,
        }
        for idx, para in enumerate(paras, 1)
    ]


def _iter_html_files(root: Path, languages: Sequence[str]) -> Iterable[Path]:
    for lang in languages:
        yield from root.joinpath(lang).glob("*.html")


# --------------------------------------------------------------------------- #
#                                   MAIN                                      #
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(description="Extract paragraphs from Wikipedia HTML dump.")
    parser.add_argument("--root", type=Path, required=True, help="Root directory containing language sub-folders.")
    parser.add_argument("--langs", nargs="+", required=True, help="Languages to process (folder names).")
    parser.add_argument("-o", "--output", type=Path, default=Path("paragraphs.jsonl"), help="Output JSONL file path.")
    parser.add_argument("-j", "--jobs", type=int, default=None, help="Parallel workers (default: CPU cores).")
    args = parser.parse_args()

    files = list(_iter_html_files(args.root, args.langs))
    logging.info("Found %d HTML files under %s", len(files), args.root)

    with args.output.open("wb") as fout, cf.ProcessPoolExecutor(max_workers=args.jobs) as ex:
        futures = {ex.submit(_process_file, f): f for f in files}
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Parsing", unit="file"):
            try:
                rows = fut.result()
                for row in rows:
                    fout.write(orjson.dumps(row) + b"\n")
            except Exception as exc:  # noqa: BLE001
                logging.warning("Error processing %s: %s", futures[fut], exc)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    main()