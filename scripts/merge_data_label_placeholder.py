# -*- coding: utf-8 -*-
"""
merge_label_mappings.py
-----------------------
Gộp 2 file mapping thành 1 bảng 2 cột (label, anchor_key) để thay thế placeholder trong dữ liệu gốc.
- Ưu tiên mapping từ file thủ công/đã duyệt: mapped_labels.csv
- Bổ sung mapping từ auto_cluster_mapped_labels.csv khi label chưa có
- Ghi thêm báo cáo conflicts (tuỳ chọn): những label xuất hiện ở cả 2 file nhưng anchor_key KHÁC nhau

Cách dùng (khai báo trực tiếp đường dẫn bên dưới, không cần argparse/environment):
    python merge_label_mappings.py
"""
from __future__ import annotations

import os
import pandas as pd

# ====== KHAI BÁO ĐƯỜNG DẪN (chỉnh theo nhu cầu) ======
PRIMARY_CSV   = "/kaggle/working/mapped_out/mapped_labels.csv"              # file đã duyệt
SECONDARY_CSV = "/kaggle/working/unassigned_autoclass/auto_cluster_mapped_labels.csv"  # file auto
OUTPUT_CSV    = "/kaggle/working/label_anchor_key_mapping.csv"              # file gộp 2 cột
WRITE_CONFLICTS = True
CONFLICTS_CSV   = "/kaggle/working/label_anchor_conflicts.csv"
# ======================================================

def read_mapping(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
    if "label" not in df.columns or "anchor_key" not in df.columns:
        raise SystemExit(f"'{path}' phải có cột 'label' và 'anchor_key'.")
    # chỉ giữ 2 cột, strip đầu/cuối
    out = df[["label", "anchor_key"]].copy()
    out["label"] = out["label"].astype(str).str.strip()
    out["anchor_key"] = out["anchor_key"].astype(str).str.strip()
    # loại bỏ dòng label rỗng
    out = out[out["label"] != ""]
    return out

def main():
    prim = read_mapping(PRIMARY_CSV)
    sec  = read_mapping(SECONDARY_CSV)

    # Drop duplicate rows in từng file
    prim = prim.drop_duplicates(subset=["label"], keep="first")
    sec  = sec.drop_duplicates(subset=["label"], keep="first")

    # Merge logic:
    # - build dict từ primary
    mapping = dict(zip(prim["label"], prim["anchor_key"]))

    # - ghi nhận conflicts khi secondary có label trùng nhưng anchor khác
    conflicts = []
    added_from_secondary = 0

    for label, key in zip(sec["label"], sec["anchor_key"]):
        if label in mapping:
            if mapping[label] != key:
                conflicts.append({"label": label, "primary_anchor_key": mapping[label], "secondary_anchor_key": key})
        else:
            mapping[label] = key
            added_from_secondary += 1

    # Xuất mapping 2 cột
    out_df = pd.DataFrame(sorted(mapping.items(), key=lambda x: (x[0].lower(), x[1].lower())), columns=["label", "anchor_key"])
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_CSV)), exist_ok=True)
    out_df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")

    print(f"Primary rows: {len(prim)}")
    print(f"Secondary rows: {len(sec)}")
    print(f"Added from secondary: {added_from_secondary}")
    print(f"Final unique labels: {len(out_df)}")
    print(f"Wrote: {OUTPUT_CSV}")

    # Xuất báo cáo conflicts (nếu bật)
    if WRITE_CONFLICTS:
        conf_df = pd.DataFrame(conflicts)
        os.makedirs(os.path.dirname(os.path.abspath(CONFLICTS_CSV)), exist_ok=True)
        conf_df.to_csv(CONFLICTS_CSV, index=False, encoding="utf-8-sig", lineterminator="\n")
        print(f"Conflicts: {len(conf_df)} (nếu >0, xem file: {CONFLICTS_CSV})")

if __name__ == "__main__":
    main()