#!/usr/bin/env python3
"""
ParlSpeech V2 — Cross-Parliament Comparative Summary
=====================================================
Produces aggregate statistics and charts across all 9 parliamentary corpora.

Usage (Blind SANE):
    python3 script.py -i <input-dir> -o <output-dir> -t <temp-dir>

Outputs (written to <output-dir>):
    summary.csv           — one row per parliament, all statistics
    chart_speeches.png    — total speeches per parliament
    chart_speakers.png    — unique speakers per parliament
    chart_avg_words.png   — average speech length per parliament
    chart_parties.png     — number of parties per parliament
    report.html           — self-contained HTML report

Source: Rauh & Schwalbach (2020), Harvard Dataverse, DOI: 10.7910/DVN/L4OAKN
"""

import argparse
import os
import sys
import time
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import rdata

# ---------------------------------------------------------------------------
# Filename-to-country mapping
# ---------------------------------------------------------------------------
COUNTRY_MAP = {
    "Corp_Bundestag_V2":       "Germany",
    "Corp_Congreso_V2":        "Spain",
    "Corp_Folketing_V2":       "Denmark",
    "Corp_HouseOfCommons_V2":  "United Kingdom",
    "Corp_Nationalrat_V2":     "Austria",
    "Corp_NZHoR_V2":           "New Zealand",
    "Corp_PSP_V2":             "Czech Republic",
    "Corp_Riksdagen_V2":       "Sweden",
    "Corp_TweedeKamer_V2":     "Netherlands",
}

# Preferred order for display
DISPLAY_ORDER = [
    "United Kingdom", "Germany", "Sweden", "Netherlands",
    "New Zealand", "Denmark", "Austria", "Spain", "Czech Republic",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Cross-parliament comparative summary")
    p.add_argument("-i", "--input",  default="/source/",  help="Input directory containing .rds files")
    p.add_argument("-o", "--output", default="/results/", help="Output directory for results")
    p.add_argument("-t", "--temp",   default="/tmp/",     help="Temporary directory")
    return p.parse_args()


def load_corpus(path: str) -> pd.DataFrame:
    """Read an RDS file and return its dataframe.

    Uses rdata's parser with manual column extraction to avoid
    numpy's fixed-width string allocation (which can exceed 100 GB).
    """
    parsed = rdata.parser.parse_file(path)
    obj = parsed.object

    # --- extract column names from pairlist attributes ---
    col_names: list[str] = []
    pair = obj.attributes
    while pair and pair.value:
        tag_name = None
        if pair.tag and pair.tag.value and hasattr(pair.tag.value, "value"):
            raw = pair.tag.value.value
            if isinstance(raw, bytes):
                tag_name = raw.decode("utf-8", errors="replace")
        if tag_name == "names":
            col_names = [
                x.value.decode("utf-8", errors="replace")  # type: ignore[union-attr]
                for x in pair.value[0].value
            ]
        rest = pair.value[1] if len(pair.value) > 1 else None
        if rest is None or rest.info.type.name == "NILVALUE":
            break
        pair = rest

    # --- convert each column to a Python-native list/array ---
    columns: dict[str, Any] = {}
    for i, col_obj in enumerate(obj.value):
        name = col_names[i] if i < len(col_names) else f"col_{i}"
        ctype = col_obj.info.type.name

        if ctype == "STR":
            columns[name] = pd.array(
                [
                    v.value.decode("utf-8", errors="replace")
                    if v.value is not None and isinstance(v.value, bytes)
                    else None
                    for v in col_obj.value
                ],
                dtype=pd.StringDtype(),
            )
        elif ctype == "INT":
            columns[name] = np.array(col_obj.value, dtype="int64")
        elif ctype == "REAL":
            columns[name] = np.array(col_obj.value, dtype="float64")
        elif ctype == "LGL":
            columns[name] = pd.array(col_obj.value, dtype=pd.BooleanDtype())
        else:
            columns[name] = list(col_obj.value)

    return pd.DataFrame(columns)


def detect_text_column(df: pd.DataFrame) -> Optional[str]:
    """Return the name of the speech-text column, or None if not found."""
    for candidate in ("text", "Text", "speech", "Speech", "terms", "Terms"):
        if candidate in df.columns:
            return candidate
    # Fall back to the longest-average object column
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        avg_len: dict[str, float] = {
            c: float(df[c].dropna().str.len().mean() or 0.0) for c in obj_cols
        }
        return max(avg_len, key=lambda k: avg_len[k])
    return None


def compute_stats(df: pd.DataFrame, country: str) -> dict[str, Any]:
    """Compute summary statistics for one corpus."""

    # --- date -----------------------------------------------------------
    if "date" in df.columns:
        date_idx = pd.DatetimeIndex(pd.to_datetime(df["date"], errors="coerce"))
        year_min = int(date_idx.year.min() or 0)
        year_max = int(date_idx.year.max() or 0)
        date_range = f"{year_min}–{year_max}"
        n_years = year_max - year_min + 1
    else:
        date_range, year_min, year_max, n_years = "N/A", None, None, None

    # --- word count -------------------------------------------------------
    # Prefer the precomputed 'terms' column (numeric word count) if present;
    # otherwise fall back to splitting the text column.
    if "terms" in df.columns and pd.api.types.is_numeric_dtype(df["terms"]):
        word_counts = df["terms"].dropna().astype(float)
        avg_words    = round(float(word_counts.mean() or 0.0), 1)
        median_words = round(float(word_counts.median() or 0.0), 1)
    else:
        text_col = detect_text_column(df)
        if text_col:
            word_counts = df[text_col].fillna("").astype(str).str.split().str.len().astype(float)
            avg_words    = round(float(word_counts.mean() or 0.0), 1)
            median_words = round(float(word_counts.median() or 0.0), 1)
        else:
            avg_words = median_words = None

    # --- speakers -------------------------------------------------------
    speaker_col = next((c for c in ("speaker", "Speaker", "name") if c in df.columns), None)
    n_speakers = df[speaker_col].nunique() if speaker_col else None

    # --- parties --------------------------------------------------------
    party_col = next((c for c in ("party", "Party", "group") if c in df.columns), None)
    if party_col:
        n_parties    = df[party_col].nunique()
        top_party    = df[party_col].value_counts().index[0]
        top_party_n  = int(df[party_col].value_counts().iloc[0])
        top_party_pct = round(top_party_n / len(df) * 100, 1)
    else:
        n_parties = top_party = top_party_n = top_party_pct = None

    return {
        "Country":              country,
        "Total Speeches":       len(df),
        "Date Range":           date_range,
        "Start Year":           int(year_min) if year_min is not None else None,
        "End Year":             int(year_max) if year_max is not None else None,
        "Years Covered":        n_years,
        "Unique Speakers":      n_speakers,
        "Unique Parties":       n_parties,
        "Avg Words/Speech":     avg_words,
        "Median Words/Speech":  median_words,
        "Most Active Party":    top_party,
        "Top Party Share (%)":  top_party_pct,
    }


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

PALETTE = {
    "speeches": "#2471A3",
    "speakers": "#E67E22",
    "words":    "#1E8449",
    "parties":  "#7D3C98",
}


def bar_chart(df: pd.DataFrame, x_col: str, y_col: str,
              title: str, xlabel: str, color: str, path: str,
              fmt_thousands: bool = True):
    ordered = df.set_index(x_col).reindex(
        [c for c in DISPLAY_ORDER if c in df[x_col].values]
    ).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(ordered[x_col], ordered[y_col], color=color, edgecolor="white")

    # Value labels
    for bar in bars:
        w = bar.get_width()
        label = f"{w:,.0f}" if fmt_thousands else f"{w:,.1f}"
        ax.text(w * 1.005, bar.get_y() + bar.get_height() / 2,
                label, va="center", ha="left", fontsize=9)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel("")
    if fmt_thousands:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.15)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def save_html(summary: pd.DataFrame, output_dir: str):
    display_cols = [
        "Country", "Total Speeches", "Date Range",
        "Unique Speakers", "Unique Parties",
        "Avg Words/Speech", "Median Words/Speech",
        "Most Active Party", "Top Party Share (%)",
    ]
    table_html = (
        summary[display_cols]
        .to_html(index=False, classes="data-table", border=0)
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>ParlSpeech V2 — Cross-Parliament Summary</title>
  <style>
    body  {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 40px auto; color: #222; }}
    h1   {{ color: #1a5276; }}
    h2   {{ color: #2471a3; margin-top: 40px; }}
    p.meta {{ font-size: 0.9em; color: #666; }}
    .data-table {{ border-collapse: collapse; width: 100%; font-size: 0.92em; }}
    .data-table th {{ background: #1a5276; color: #fff; padding: 8px 12px; text-align: left; }}
    .data-table td {{ padding: 7px 12px; border-bottom: 1px solid #ddd; }}
    .data-table tr:nth-child(even) {{ background: #f4f6f7; }}
    img {{ max-width: 100%; margin: 10px 0 30px; border: 1px solid #ddd; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>ParlSpeech V2 — Cross-Parliament Comparative Summary</h1>
  <p class="meta">
    Source: Rauh &amp; Schwalbach (2020), Harvard Dataverse,
    DOI: <a href="https://doi.org/10.7910/DVN/L4OAKN">10.7910/DVN/L4OAKN</a>
    &nbsp;|&nbsp; Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}
  </p>

  <h2>Summary Table</h2>
  {table_html}

  <h2>Total Speeches by Parliament</h2>
  <img src="chart_speeches.png" alt="Total speeches">

  <h2>Unique Speakers by Parliament</h2>
  <img src="chart_speakers.png" alt="Unique speakers">

  <h2>Average Speech Length (words) by Parliament</h2>
  <img src="chart_avg_words.png" alt="Average words per speech">

  <h2>Number of Parties by Parliament</h2>
  <img src="chart_parties.png" alt="Number of parties">
</body>
</html>"""

    with open(os.path.join(output_dir, "report.html"), "w", encoding="utf-8") as f:
        f.write(html)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.temp,   exist_ok=True)

    rds_files = sorted(
        f for f in os.listdir(args.input) if f.lower().endswith(".rds")
    )

    if not rds_files:
        print("ERROR: No .rds files found in input directory.", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(rds_files)} corpus file(s). Processing...\n")

    rows = []
    for fname in rds_files:
        stem    = fname[:-4]          # strip .rds
        country = COUNTRY_MAP.get(stem, stem)
        path    = os.path.join(args.input, fname)

        print(f"[{country}] Loading {fname} ...")
        t0 = time.time()
        df = load_corpus(path)
        print(f"[{country}] {len(df):,} rows loaded in {time.time() - t0:.1f}s")

        stats = compute_stats(df, country)
        rows.append(stats)

        del df   # release memory before loading the next file

    summary = pd.DataFrame(rows)

    # Reorder rows for display
    cat = pd.CategoricalDtype(categories=DISPLAY_ORDER, ordered=True)
    summary["_order"] = pd.Categorical(summary["Country"], dtype=cat)
    summary = summary.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    # --- CSV ---
    summary.to_csv(os.path.join(args.output, "summary.csv"), index=False)
    print("\nSaved summary.csv")

    # --- Charts ---
    bar_chart(summary, "Country", "Total Speeches",
              "Total Speeches by Parliament", "Number of speeches",
              PALETTE["speeches"], os.path.join(args.output, "chart_speeches.png"))

    bar_chart(summary, "Country", "Unique Speakers",
              "Unique Speakers by Parliament", "Number of distinct speakers",
              PALETTE["speakers"], os.path.join(args.output, "chart_speakers.png"))

    bar_chart(summary, "Country", "Avg Words/Speech",
              "Average Speech Length by Parliament", "Mean words per speech",
              PALETTE["words"], os.path.join(args.output, "chart_avg_words.png"),
              fmt_thousands=False)

    bar_chart(summary, "Country", "Unique Parties",
              "Number of Parties by Parliament", "Distinct parties in corpus",
              PALETTE["parties"], os.path.join(args.output, "chart_parties.png"))

    print("Saved 4 charts")

    # --- HTML ---
    save_html(summary, args.output)
    print("Saved report.html")

    # --- Console summary ---
    print("\n" + "=" * 65)
    print("CROSS-PARLIAMENT SUMMARY")
    print("=" * 65)
    print(summary[[
        "Country", "Total Speeches", "Date Range",
        "Unique Speakers", "Avg Words/Speech"
    ]].to_string(index=False))
    print("=" * 65)
    print(f"\nAll results written to: {args.output}")


if __name__ == "__main__":
    main()
