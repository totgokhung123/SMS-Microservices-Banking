# -*- coding: utf-8 -*-
"""
extract_labels_from_response_vn.py
----------------------------------
Xuất danh sách các nhãn dạng {{...}} (ví dụ: {{Credit Card}}, {{Cập nhật}}) từ cột "response"/"responses".
- Hỗ trợ tiếng Việt (UTF-8 BOM để tránh lỗi khi mở Excel).
- Không dùng argparse; cấu hình bằng biến phía dưới hoặc biến môi trường.
- Tạo 2 file đầu ra:
    1) labels_unique.csv  : Danh sách nhãn duy nhất + số lần xuất hiện.
    2) labels_occurrences.csv : Tất cả lần xuất hiện, gồm: hàng, cột, nhãn, vị trí start/end, snippet ngắn.

Cách dùng:
1) Mở file và chỉnh các biến cấu hình ở dưới (hoặc đặt biến môi trường CSV_INPUT, CSV_UNIQUE, CSV_OCC, CSV_COLUMN).
2) Chạy:  python extract_labels_from_response_vn.py
"""
from __future__ import annotations

import os
import re
import sys
from typing import List, Tuple, Dict

import pandas as pd


# ----------------- CẤU HÌNH (chỉnh ở đây) -----------------
INPUT_CSV   = "/kaggle/input/data-final-lay-nhan/final_sua.csv"   # ví dụ: r"D:\data\input.csv"
OUTPUT_UNIQUE = "/kaggle/working/labels_unique.csv" # ví dụ: r"D:\data\labels_unique.csv"
OUTPUT_OCC    = "/kaggle/working/labels_occurrences.csv" # ví dụ: r"D:\data\labels_occurrences.csv"
COLUMN_NAME   = "response" # để trống sẽ tự tìm "response" hoặc "responses"

# Cho phép override qua biến môi trường nếu muốn
INPUT_CSV     = os.environ.get("CSV_INPUT",  INPUT_CSV)
OUTPUT_UNIQUE = os.environ.get("CSV_UNIQUE", OUTPUT_UNIQUE)
OUTPUT_OCC    = os.environ.get("CSV_OCC",    OUTPUT_OCC)
COLUMN_NAME   = os.environ.get("CSV_COLUMN", COLUMN_NAME)


# --------- Regex & helpers ---------
# Regex match {{ ... }} với nội dung KHÔNG chứa { } hay xuống dòng để tránh match lố
# Cho phép 1..200 ký tự bên trong để hạn chế tham lam
PLACEHOLDER_RE = re.compile(r"\{\{\s*([^\{\}\n\r]{1,200}?)\s*\}\}", flags=re.UNICODE)

def normalize_whitespace(s: str) -> str:
    return re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()

def extract_labels(text: str) -> List[Tuple[str, int, int, str]]:
    """
    Trả về list tuple: (label, start, end, snippet)
    - label: nội dung bên trong {{ ... }} đã strip & gộp khoảng trắng
    - start, end: vị trí trong chuỗi gốc
    - snippet: đoạn ngắn quanh match để kiểm tra (±30 ký tự)
    """
    res = []
    if not isinstance(text, str):
        return res

    for m in PLACEHOLDER_RE.finditer(text):
        inner = normalize_whitespace(m.group(1))
        start, end = m.start(), m.end()
        # tạo snippet an toàn cho Unicode
        left = max(0, start - 30)
        right = min(len(text), end + 30)
        snippet = text[left:right].replace("\n", " ").replace("\r", " ")
        res.append((inner, start, end, snippet))
    return res


def main():
    global INPUT_CSV, OUTPUT_UNIQUE, OUTPUT_OCC, COLUMN_NAME

    if not INPUT_CSV:
        INPUT_CSV = input("Nhập đường dẫn CSV đầu vào: ").strip()
    if not OUTPUT_UNIQUE:
        OUTPUT_UNIQUE = input("Nhập đường dẫn file nhãn duy nhất (labels_unique.csv): ").strip()
    if not OUTPUT_OCC:
        OUTPUT_OCC = input("Nhập đường dẫn file tất cả lần xuất hiện (labels_occurrences.csv): ").strip()

    if not INPUT_CSV or not OUTPUT_UNIQUE or not OUTPUT_OCC:
        print("Thiếu đường dẫn. Hãy chỉnh trong file hoặc đặt biến môi trường CSV_INPUT / CSV_UNIQUE / CSV_OCC.")
        sys.exit(2)

    # Đọc CSV với UTF-8 BOM (fallback utf-8)
    try:
        df = pd.read_csv(INPUT_CSV, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, dtype=str, encoding="utf-8")

    # Xác định cột dữ liệu
    cols_lower = {c.lower(): c for c in df.columns}
    if not COLUMN_NAME:
        if "response" in cols_lower:
            COLUMN_NAME = cols_lower["response"]
        elif "responses" in cols_lower:
            COLUMN_NAME = cols_lower["responses"]
        else:
            print("Không tìm thấy cột 'response' hoặc 'responses' trong CSV. Vui lòng chỉ định CSV_COLUMN.")
            sys.exit(3)
    elif COLUMN_NAME not in df.columns:
        print(f"Không tìm thấy cột '{COLUMN_NAME}' trong CSV. Các cột hiện có: {list(df.columns)}")
        sys.exit(3)

    # Duyệt & trích xuất
    unique_counts: Dict[str, int] = {}
    occ_rows = []

    for idx, val in df[COLUMN_NAME].items():
        matches = extract_labels(val)
        for label, start, end, snippet in matches:
            unique_counts[label] = unique_counts.get(label, 0) + 1
            occ_rows.append({
                "row_index": idx,
                "column": COLUMN_NAME,
                "label": label,
                "start": start,
                "end": end,
                "snippet": snippet
            })

    # Ghi labels_unique.csv
    uniq_df = pd.DataFrame(
        [{"label": k, "count": v} for k, v in sorted(unique_counts.items(), key=lambda x: (-x[1], x[0]))]
    )
    # Ghi labels_occurrences.csv
    occ_df = pd.DataFrame(occ_rows)

    # Tạo thư mục nếu cần
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_UNIQUE)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_OCC)), exist_ok=True)

    uniq_df.to_csv(OUTPUT_UNIQUE, index=False, encoding="utf-8-sig", lineterminator="\n")
    occ_df.to_csv(OUTPUT_OCC, index=False, encoding="utf-8-sig", lineterminator="\n")

    print("Hoàn tất.")
    print("Nhãn duy nhất:", OUTPUT_UNIQUE)
    print("Danh sách lần xuất hiện:", OUTPUT_OCC)


if __name__ == "__main__":
    main()