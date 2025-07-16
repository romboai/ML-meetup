from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

import pandas as pd

__all__ = ["load_paragraphs", "load_questions"]

def _page_from_url(url: str) -> str:
    """Return page title from a full Wikipedia URL."""
    path = urlparse(url).path
    title = path.split("/wiki/")[-1].split("#")[0].strip("/")
    return unquote(title).replace("_", " ")

def load_paragraphs(data_dir: str | Path = ".", lang: str = "sc") -> pd.DataFrame:

    data_dir = Path(data_dir)
    items = []

    with open(data_dir / "paragraphs.jsonl", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if obj.get("lang") == lang:
                items.append(obj)

    df = pd.DataFrame(items)
    df["page_cf"] = df["page_title"].str.casefold()

    print(f"Loaded {len(df)} paragraphs in '{lang}'")
    return df


def load_questions(data_dir: str | Path = ".", lang: str = 'sc') -> pd.DataFrame:

    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "nq_sc_sc.csv")
    if lang == 'sc':
        df = df[~df["question_sc"].isna()].copy()
        df["gt_page"] = df["url_sc"].apply(_page_from_url)
    elif lang == 'en':
        df = df[~df["question"].isna()].copy()
        df["gt_page"] = df["url_en"].apply(_page_from_url)
    
    df["page_cf"] = df["gt_page"].str.casefold()

    print(f"Loaded {len(df)} questions")
    return df