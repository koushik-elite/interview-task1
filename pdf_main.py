import json
import re
import pdfplumber
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

DATA_DIR   = Path("data")          # folder containing all 10-Q PDFs
OUTPUT_DIR = Path("output")        # folder where JSON files are saved


# ── FILENAME PARSER ───────────────────────────────────────────────────────────

def parse_filename(stem: str) -> dict:
    """
    Split a filename like "2022 Q3 AAPL" into structured metadata.

    Handles these naming patterns:
      "2022 Q3 AAPL"        → year=2022  quarter=Q3  ticker=AAPL
      "2022 Q3 AAPL 10Q"    → same  (ignores trailing tokens like 10Q)
      "AAPL Q3 2022"        → same  (order-independent)
      "2023_Q1_MSFT"        → same  (underscore separator)

    Returns:
      { "year": "2022", "quarter": "Q3", "ticker": "AAPL",
        "period": "Q3_2022", "source_file": "2022 Q3 AAPL.pdf" }
    """
    # Normalise separators → spaces, then split into tokens
    tokens = re.split(r"[\s_\-]+", stem.strip())

    year    = None
    quarter = None
    ticker  = None

    known_tickers = {
        "AAPL", "AMZN", "INTC", "MSFT", "NVDA",
        "GOOGL", "GOOG", "META", "TSLA", "AMD",
    }

    for token in tokens:
        t = token.upper()

        # Year  — 4-digit number starting with 19xx or 20xx
        if re.fullmatch(r"(19|20)\d{2}", t):
            year = t

        # Quarter  — Q1 / Q2 / Q3 / Q4
        elif re.fullmatch(r"Q[1-4]", t):
            quarter = t

        # Ticker  — known list OR all-uppercase 1-5 alpha chars
        elif t in known_tickers or re.fullmatch(r"[A-Z]{1,5}", t):
            # Skip generic tokens that look like tickers but aren't
            if t not in {"Q", "10Q", "10K", "SEC", "PDF", "FORM"}:
                ticker = t

    # Derive a clean period label
    period = f"{quarter}_{year}" if quarter and year else "UNKNOWN"

    return {
        "year":        year    or "UNKNOWN",
        "quarter":     quarter or "UNKNOWN",
        "ticker":      ticker  or "UNKNOWN",
        "period":      period,
    }


# ── TABLE HELPERS ─────────────────────────────────────────────────────────────

def is_inside_table(word_top, word_bottom, table_bboxes):
    """
    Return True if a word's vertical span overlaps any table bounding box.
    Prevents text inside table cells from being duplicated as a text block.
    """
    for (x0, top, x1, bottom) in table_bboxes:
        if word_top < bottom - 2 and word_bottom > top + 2:
            return True
    return False


def table_to_markdown(raw_table):
    """
    Convert pdfplumber raw table (list of lists) → markdown string.
    Pure Python — zero external dependencies.
    """
    if not raw_table or len(raw_table) < 2:
        return None

    # Clean cells
    cleaned = [
        [cell.strip() if cell else "" for cell in row]
        for row in raw_table
    ]

    col_count = max(len(row) for row in cleaned)
    padded    = [row + [""] * (col_count - len(row)) for row in cleaned]
    col_widths = [
        max(len(row[c]) for row in padded)
        for c in range(col_count)
    ]

    def fmt_row(row):
        return "| " + " | ".join(row[c].ljust(col_widths[c]) for c in range(col_count)) + " |"

    def sep_row():
        return "| " + " | ".join("-" * w for w in col_widths) + " |"

    lines = [fmt_row(padded[0]), sep_row()]
    for row in padded[1:]:
        lines.append(fmt_row(row))

    return "\n".join(lines)


# ── PAGE EXTRACTOR ────────────────────────────────────────────────────────────

def extract_page_elements(page):
    """
    Extract all text blocks and tables from one page in reading order.

    Returns list of { type, y_top, content, meta }
    sorted by y_top (top-to-bottom).
    """
    elements    = []
    page_height = page.height
    table_bboxes = []

    # ── Tables first (need their bboxes to filter text) ───────────────────────
    for table in page.find_tables():
        raw  = table.extract()
        bbox = table.bbox
        md   = table_to_markdown(raw)

        if md:
            table_bboxes.append(bbox)
            _, top, _, _ = bbox
            elements.append({
                "type":    "table",
                "y_top":   top,
                "content": md,
                "meta": {
                    "rows": len(raw) - 1,
                    "cols": len(raw[0]) if raw else 0,
                },
            })

    # ── Words → lines → paragraphs (skipping table regions) ──────────────────
    words = page.extract_words(keep_blank_chars=False, use_text_flow=True)

    if words:
        header_cutoff = page_height * 0.05
        footer_cutoff = page_height * 0.95

        filtered = [
            w for w in words
            if not is_inside_table(w["top"], w["bottom"], table_bboxes)
            and w["top"]    > header_cutoff
            and w["bottom"] < footer_cutoff
        ]

        # Words → lines (4px vertical tolerance)
        lines, current_line, prev_top = [], [], None
        for word in filtered:
            if prev_top is None or abs(word["top"] - prev_top) <= 4:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(current_line)
                current_line = [word]
            prev_top = word["top"]
        if current_line:
            lines.append(current_line)

        # Lines → paragraphs (12px gap between paragraphs)
        paragraphs, current_para, prev_bottom = [], [], None
        for line in lines:
            line_top    = min(w["top"]    for w in line)
            line_bottom = max(w["bottom"] for w in line)
            if prev_bottom is None or (line_top - prev_bottom) <= 12:
                current_para.append(line)
            else:
                if current_para:
                    paragraphs.append(current_para)
                current_para = [line]
            prev_bottom = line_bottom
        if current_para:
            paragraphs.append(current_para)

        # Paragraphs → text elements
        for para in paragraphs:
            all_words = [w for line in para for w in line]
            text  = " ".join(w["text"] for w in all_words).strip()
            y_top = min(w["top"] for w in all_words)
            if len(text) < 10:
                continue
            elements.append({
                "type":    "text",
                "y_top":   y_top,
                "content": text,
                "meta":    {},
            })

    # Sort by y position → correct reading order
    elements.sort(key=lambda e: e["y_top"])
    return elements


# ── SINGLE FILE PROCESSOR ─────────────────────────────────────────────────────

def process_pdf(pdf_path: Path) -> dict:
    """
    Process one PDF file end-to-end.
    Returns the full document JSON dict (also saves it to OUTPUT_DIR).
    """
    # Parse metadata from filename
    meta = parse_filename(pdf_path.stem)

    print(f"\n{'─'*60}")
    print(f"  File    : {pdf_path.name}")
    print(f"  Ticker  : {meta['ticker']}")
    print(f"  Year    : {meta['year']}")
    print(f"  Quarter : {meta['quarter']}")
    print(f"  Period  : {meta['period']}")
    print(f"{'─'*60}")

    document_elements = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            elements = extract_page_elements(page)

            for pos, el in enumerate(elements, start=1):
                document_elements.append({
                    # ── document-level metadata (from filename) ──
                    "ticker":      meta["ticker"],
                    "year":        meta["year"],
                    "quarter":     meta["quarter"],
                    "period":      meta["period"],
                    "source_file": pdf_path.name,
                    # ── element-level metadata ───────────────────
                    "type":        el["type"],
                    "page":        page_num,
                    "position":    pos,
                    "content":     el["content"],
                    **el["meta"],  # rows + cols for tables
                })

            n_text  = sum(1 for e in elements if e["type"] == "text")
            n_table = sum(1 for e in elements if e["type"] == "table")
            print(f"  Page {page_num:3d}/{total_pages}"
                  f"  →  {n_text:3d} text  |  {n_table} tables")

    output = {
        "metadata": {
            "ticker":         meta["ticker"],
            "year":           meta["year"],
            "quarter":        meta["quarter"],
            "period":         meta["period"],
            "source_file":    pdf_path.name,
            "total_pages":    total_pages,
            "total_elements": len(document_elements),
            "text_count":     sum(1 for e in document_elements if e["type"] == "text"),
            "table_count":    sum(1 for e in document_elements if e["type"] == "table"),
        },
        "elements": document_elements,
    }

    # Save individual JSON  e.g. output/2022_Q3_AAPL.json
    out_file = OUTPUT_DIR / f"{meta['year']}_{meta['quarter']}_{meta['ticker']}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    m = output["metadata"]
    print(f"\n  ✓ {m['total_elements']} elements  "
          f"({m['text_count']} text / {m['table_count']} tables)"
          f"  →  saved {out_file.name}")

    return output


# ── MAIN — processes every PDF in data/ ──────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all PDFs in the data directory
    pdf_files = sorted(DATA_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in '{DATA_DIR}/'")
        print("Put your 10-Q PDFs there and re-run.")
        return

    print(f"{'='*60}")
    print(f"  Found {len(pdf_files)} PDF(s) in '{DATA_DIR}/'")
    print(f"{'='*60}")

    all_results = []
    failed      = []

    for pdf_path in pdf_files:
        try:
            result = process_pdf(pdf_path)
            all_results.append(result["metadata"])
        except Exception as e:
            print(f"\n  ✗ FAILED: {pdf_path.name}  →  {e}")
            failed.append(pdf_path.name)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE — {len(all_results)} succeeded / {len(failed)} failed")
    print(f"{'='*60}\n")

    print(f"  {'FILE':<30} {'TICKER':<8} {'YEAR':<6} {'QTR':<5} {'ELEMENTS':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*6} {'-'*5} {'-'*8}")
    for m in all_results:
        print(f"  {m['source_file']:<30} {m['ticker']:<8} {m['year']:<6}"
              f" {m['quarter']:<5} {m['total_elements']:>8}")

    if failed:
        print(f"\n  Failed files:")
        for f in failed:
            print(f"    ✗ {f}")

    print(f"\n  JSON files saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()