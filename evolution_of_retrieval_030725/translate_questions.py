"""translate_questions.py
Translate English questions in a CSV file to Sardinian.

Two strategies are available:
    * **pivot**   : English → Italian → Sardinian (NLLB‑200 + Apertium ita‑srd)
    * **direct**  : English → Sardinian (NLLB‑200 eng_Latn→srd_Latn)

The script can run either strategy or *both* in one pass and can resume from a
previously generated output file without re‑translating completed rows.

Example
-------
$ python translate_questions.py nq_sc.csv  # generates nq_sc_sc.csv with both
$ python translate_questions.py nq_sc.csv -m direct  # only direct column

Dependencies
------------
* transformers ≥ 4.38, sentencepiece, accelerate
* pandas, tqdm
* Apertium + pair *ita‑srd* (sudo apt install apertium-ita-srd)
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
from typing import List, Literal

import pandas as pd
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Translation helpers
# ---------------------------------------------------------------------------

def _build_nllb_pipe(tgt_lang: str, *, device: int = 0):
    """Return an NLLB‑200 pipeline.

    Args:
        tgt_lang: Target ISO code in NLLB tag format (e.g. ``ita_Latn``).
        device: CUDA index (‑1 for CPU).

    Returns:
        transformers.Pipeline: Configured for eng_Latn → *tgt_lang*.
    """
    model_name = "facebook/nllb-200-distilled-600M"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    return pipeline(
        "translation",
        model=mdl,
        tokenizer=tok,
        src_lang="eng_Latn",
        tgt_lang=tgt_lang,
        device=device,
        max_length=512,
    )


def _apertium_translate(text: str, pair: str = "ita-srd") -> str:
    """Translate *text* via Apertium rule‑based engine."""
    proc = subprocess.run(
        ["apertium", pair],
        input=text.encode(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout.decode().strip()


# ---------------------------------------------------------------------------
# Core routine
# ---------------------------------------------------------------------------

def _translate_batch(
    df: pd.DataFrame,
    mode: Literal["direct", "pivot", "both"],
    device: int,
) -> pd.DataFrame:
    """Translate questions per *mode* filling missing cells only."""
    need_direct = mode in {"direct", "both"}
    need_pivot = mode in {"pivot", "both"}

    pipe_direct = _build_nllb_pipe("srd_Latn", device=device) if need_direct else None
    pipe_ita = _build_nllb_pipe("ita_Latn", device=device) if need_pivot else None

    for idx, question in tqdm(df["question"].items(), unit="sent", desc="Translating"):
        # Direct
        if need_direct and pd.isna(df.at[idx, "question_sc"]):
            df.at[idx, "question_sc"] = pipe_direct(question, truncation=True)[0]["translation_text"]

        # Pivot
        if need_pivot and pd.isna(df.at[idx, "question_sc_pivot"]):
            it_text = pipe_ita(question, truncation=True)[0]["translation_text"]
            df.at[idx, "question_sc_pivot"] = _apertium_translate(it_text)

        # Flush to disk periodically (every 100 rows)
        if idx % 100 == 0:
            yield df

    yield df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli() -> None:
    parser = argparse.ArgumentParser(description="Translate English questions to Sardinian (direct, pivot, or both).")
    parser.add_argument(
        "input_csv",
        type=Path,
        nargs="?",
        default=Path("nq_sc.csv"),
        help="Input CSV (default: nq_sc.csv).",
    )
    parser.add_argument(
        "-o",
        "--output_csv",
        type=Path,
        default=None,
        help="Output CSV (default: <input>_sc.csv).",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["pivot", "direct", "both"],
        default="both",
        help="Translation mode.",
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA index (‑1 = CPU).")
    args = parser.parse_args()

    out_path = args.output_csv or args.input_csv.with_name(args.input_csv.stem + "_sc.csv")

    # Load source
    src_df = pd.read_csv(args.input_csv)
    if "question" not in src_df.columns:
        raise ValueError("Column 'question' missing in input CSV.")

    # Initialise target columns if absent.
    cols = {"direct": "question_sc", "pivot": "question_sc_pivot"}
    if args.mode in {"direct", "both"} and cols["direct"] not in src_df.columns:
        src_df[cols["direct"]] = pd.NA
    if args.mode in {"pivot", "both"} and cols["pivot"] not in src_df.columns:
        src_df[cols["pivot"]] = pd.NA

    # Resume support
    if out_path.exists():
        dst_df = pd.read_csv(out_path)
        src_df = dst_df.combine_first(src_df)

    # Translate rows lazily and flush
    for partial_df in _translate_batch(src_df, args.mode, args.device):
        partial_df.to_csv(out_path, index=False)

    print(f"✔ Saved to {out_path}")


if __name__ == "__main__":
    _cli()