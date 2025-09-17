# ======================================================
# -*- coding: utf-8 -*-
"""
replace_placeholders_with_mapping_v2.py
---------------------------------------
Thay placeholder {{label}} trong cột "response" bằng {{ANCHOR_KEY}} dựa trên 2 nguồn mapping:

1) label_final_map.csv  (2 cột: label, anchor_key)  -> ƯU TIÊN CAO HƠN
2) placeholder_mapping_report.csv  (bạn đã thêm cột mapping) -> BỔ SUNG

Nếu vẫn không tìm thấy mapping cho một label thì **xóa dấu ngoặc** và giữ nguyên nội dung bên trong.

- Đọc/ghi UTF-8 (utf-8-sig) để an toàn tiếng Việt.
- Không cần argparse; khai báo đường dẫn ngay trong code (Kaggle-friendly).
- Tự tìm 'response' hoặc 'responses' nếu COLUMN_NAME để trống.
- Xuất thêm báo cáo tóm tắt các thay thế/loại bỏ.

Chạy:
    python replace_placeholders_with_mapping_v2.py
"""

from __future__ import annotations
import os, re, sys
from typing import Dict, Tuple, List
import pandas as pd

# ====== KHAI BÁO ĐƯỜNG DẪN (chỉnh theo nhu cầu) ======
INPUT_CSV    = "E:/HDBank_Hackathon/source/data/raw/csv/final_sua.csv"  # file dữ liệu gốc
PRIMARY_MAP = "E:/HDBank_Hackathon/source/data/raw/csv/trích xuất nhãn/processing_mapping_label/label_final_map.csv"                 # 2 cột: label, anchor_key
REPORT_MAP  = "E:/HDBank_Hackathon/source/data/raw/csv/placeholder_mapping_report.csv"      # đã bổ sung cột mapping
OUTPUT_CSV  = "E:/HDBank_Hackathon/source/data/raw/csv/final_sua_mapped_v2.csv"             # kết quả sau khi thay
SUMMARY_CSV = "E:/HDBank_Hackathon/source/data/raw/csv/placeholder_replacement_summary.csv" # nhật ký thay thế
COLUMN_NAME = ""  # để trống: tự phát hiện 'response' hoặc 'responses' (không phân biệt hoa/thường)
# ======================================================

PLACEHOLDER_RE = re.compile(r"\{\{\s*([^\{\}\n\r]{1,200}?)\s*\}\}", flags=re.UNICODE)

def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s), flags=re.UNICODE).strip()

def load_csv_utf8(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="utf-8")

def build_primary_mapping(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Đọc file 2 cột label, anchor_key (ưu tiên cao)."""
    df = load_csv_utf8(path)
    cols_lower = {c.lower(): c for c in df.columns}
    if "label" not in cols_lower or "anchor_key" not in cols_lower:
        raise SystemExit(f"'{path}' phải có cột 'label' và 'anchor_key'. Cột hiện có: {list(df.columns)}")
    df = df[[cols_lower["label"], cols_lower["anchor_key"]]].copy()
    df.columns = ["label","anchor_key"]
    df["label"] = df["label"].apply(normalize_ws)
    df["anchor_key"] = df["anchor_key"].apply(normalize_ws)
    df = df[(df["label"]!="") & (df["anchor_key"]!="")].drop_duplicates(subset=["label"], keep="first")
    exact = dict(zip(df["label"], df["anchor_key"]))
    lower = {k.casefold(): v for k, v in exact.items()}
    return exact, lower

def detect_report_mapping_col(df: pd.DataFrame) -> str:
    """Tìm cột chứa anchor_key trong report mapping (bạn có thể đặt tên khác).
       Ưu tiên: 'anchor_key', 'mapping', 'mapped_anchor_key', 'anchor', 'placeholder'.
    """
    candidates = ["anchor_key", "mapping", "mapped_anchor_key", "anchor", "placeholder"]
    normcols = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in normcols:
            return normcols[key]
    # fallback: chọn cột thứ 2 (khác 'label' và 'occurrences'), nếu có
    others = [c for c in df.columns if c.lower() not in {"label","occurrences"}]
    if len(others) >= 1:
        return others[0]
    raise SystemExit("Không tìm thấy cột mapping trong placeholder_mapping_report.csv. Hãy thêm 'anchor_key' hoặc 'mapping'.")

def build_report_mapping(path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Đọc report đã gán thêm mapping (lấy label -> anchor_key)."""
    df = load_csv_utf8(path)
    # Tìm cột label
    cols_lower = {c.lower(): c for c in df.columns}
    if "label" not in cols_lower:
        raise SystemExit(f"'{path}' phải có cột 'label'. Các cột hiện có: {list(df.columns)}")
    label_col = cols_lower["label"]
    map_col = detect_report_mapping_col(df)
    # Chuẩn hóa
    sub = df[[label_col, map_col]].copy()
    sub.columns = ["label","anchor_key"]
    sub["label"] = sub["label"].apply(normalize_ws)
    sub["anchor_key"] = sub["anchor_key"].apply(normalize_ws)
    sub = sub[(sub["label"]!="") & (sub["anchor_key"]!="")]
    sub = sub.drop_duplicates(subset=["label"], keep="first")
    exact = dict(zip(sub["label"], sub["anchor_key"]))
    lower = {k.casefold(): v for k, v in exact.items()}
    return exact, lower

def pick_anchor(inner_text: str,
                prim_exact: Dict[str,str], prim_lower: Dict[str,str],
                rep_exact: Dict[str,str], rep_lower: Dict[str,str]) -> Tuple[str, str, str]:
    """Trả về (replacement_anchor_key, matched_label_key, source)"""
    t0 = normalize_ws(inner_text)     # giữ nguyên dấu
    # 1) Ưu tiên primary
    if t0 in prim_exact:
        return prim_exact[t0], t0, "primary"
    t1 = t0.casefold()
    if t1 in prim_lower:
        return prim_lower[t1], t0, "primary-ci"

    # 2) Report mapping (bổ sung)
    if t0 in rep_exact:
        return rep_exact[t0], t0, "report"
    if t1 in rep_lower:
        return rep_lower[t1], t0, "report-ci"

    # 3) Không có -> trả rỗng để strip braces
    return "", t0, ""

def replace_placeholders(text: str,
                         prim_exact: Dict[str,str], prim_lower: Dict[str,str],
                         rep_exact: Dict[str,str], rep_lower: Dict[str,str],
                         tally: Dict[str,int], stripped: Dict[str,int]) -> Tuple[str,int]:
    """Thay trong một ô 'response'. Nếu không tìm thấy mapping -> bỏ {{}}"""
    if not isinstance(text, str) or text == "":
        return text, 0

    replaced = 0
    out_parts = []
    last = 0

    for m in PLACEHOLDER_RE.finditer(text):
        out_parts.append(text[last:m.start()])
        inner = m.group(1)
        anchor, used_label, src = pick_anchor(inner, prim_exact, prim_lower, rep_exact, rep_lower)
        if anchor:
            out_parts.append(f"{{{{{anchor}}}}}")
            replaced += 1
            tally_key = f"{src}:{anchor}"
            tally[tally_key] = tally.get(tally_key, 0) + 1
        else:
            # strip braces -> giữ nguyên nội dung bên trong
            out_parts.append(normalize_ws(inner))
            stripped[normalize_ws(inner)] = stripped.get(normalize_ws(inner), 0) + 1
        last = m.end()

    out_parts.append(text[last:])
    return "".join(out_parts), replaced

def main():
    # 1) Load mappings
    prim_exact, prim_lower = build_primary_mapping(PRIMARY_MAP)
    rep_exact,  rep_lower  = build_report_mapping(REPORT_MAP)

    # 2) Load data
    try:
        df = pd.read_csv(INPUT_CSV, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, dtype=str, encoding="utf-8")

    # 3) Column detection
    global COLUMN_NAME
    if not COLUMN_NAME:
        lowers = {c.lower(): c for c in df.columns}
        if "response" in lowers: COLUMN_NAME = lowers["response"]
        elif "responses" in lowers: COLUMN_NAME = lowers["responses"]
        else:
            print("Không tìm thấy cột 'response' hoặc 'responses'. Các cột:", list(df.columns))
            sys.exit(3)
    elif COLUMN_NAME not in df.columns:
        print(f"Không tìm thấy cột '{COLUMN_NAME}'. Các cột:", list(df.columns))
        sys.exit(3)

    # 4) Replace / Strip
    tally = {}    # đếm theo nguồn + anchor
    stripped = {} # đếm nhãn bị bỏ ngoặc
    total_replaced = 0

    def _do(x):
        nonlocal total_replaced
        new_text, cnt = replace_placeholders(
            x, prim_exact, prim_lower, rep_exact, rep_lower, tally, stripped
        )
        total_replaced += cnt
        return new_text

    df[COLUMN_NAME] = df[COLUMN_NAME].apply(_do)

    # 5) Write outputs
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")

    # 6) Summary
    rows = []
    for k, v in sorted(tally.items(), key=lambda x: (-x[1], x[0])):
        src, anchor = k.split(":", 1)
        rows.append({"action": "mapped", "source": src, "anchor_key": anchor, "occurrences": v})
    for lbl, v in sorted(stripped.items(), key=lambda x: (-x[1], x[0])):
        rows.append({"action": "stripped", "label": lbl, "occurrences": v})

    rep = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(SUMMARY_CSV)), exist_ok=True)
    rep.to_csv(SUMMARY_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")

    print(f"Đã thay {total_replaced} placeholder. File kết quả: {OUTPUT_CSV}")
    print(f"Nhật ký: {SUMMARY_CSV}")

if __name__ == "__main__":
    main()