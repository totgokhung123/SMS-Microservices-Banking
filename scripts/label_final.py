# -*- coding: utf-8 -*-
"""
map_to_allowed_placeholders.py
------------------------------
Mục tiêu: GÁN TẤT CẢ label vào danh sách placeholder KEY được CHO SẴN (không sinh thêm key).
- Ưu tiên mapping có sẵn trong Data_label_anchor_key_mapping.csv
- Tự phân loại phần còn lại vào đúng các KEY trong ALLOWED_KEYS bằng luật VN–EN + scoring
- Xuất đúng 2 cột (label, anchor_key) để dùng thay thế placeholder trong dữ liệu gốc

Đầu vào (mặc định):
- EXISTING_MAP: /mnt/data/Data_label_anchor_key_mapping.csv   (bắt buộc có cột 'label','anchor_key')
- EXTRA_LABELS: /mnt/data/unassigned_labels.csv (tùy chọn; có cột 'label','count')

Đầu ra:
- /mnt/data/final_mapping_out/label_anchor_key_final.csv  (2 cột: label, anchor_key)
- /mnt/data/final_mapping_out/conflicts.csv               (nếu có xung đột giữa existing & auto)
- /mnt/data/final_mapping_out/unmapped.csv                (label chưa phân loại được)
"""
from __future__ import annotations
import os, re, unicodedata
from typing import List, Dict, Tuple
from collections import Counter
import pandas as pd

# ====== KHAI BÁO ĐƯỜNG DẪN ======
EXISTING_MAP = "E:/HDBank_Hackathon/source/data/raw/csv/trích xuất nhãn/processing_mapping_label/Data_label_anchor_key_mapping.csv"
EXTRA_LABELS = ""  # nếu không có thì bỏ qua
OUT_DIR      = "E:/HDBank_Hackathon/source/data/raw/csv/final_mapping_out"
# =================================

# ----- Allowed placeholder keys (KHÔNG sinh key mới ngoài danh sách này) -----
ALLOWED_KEYS = [
    "SUPPORT_PHONE","SUPPORT_EMAIL","SUPPORT_WEBSITE","LIVE_CHAT_URL","WORKING_HOURS",
    "BRANCH_ADDRESS","BRANCH_LOCATOR_URL","ATM_LOCATOR_URL",
    "LOST_STOLEN_CARD_PHONE","CARD_ACTIVATION_PHONE","CARD_ACTIVATION_URL","CARD_FREEZE_UNFREEZE","MANAGE_CARDS_URL",
    "CARD_NUMBER_FIELD","CARD_PIN_FIELD","CARD_CVV_FIELD","CARD_EXPIRY_FIELD","CREDIT_LIMIT_INFO","CASH_ADVANCE_FEE",
    "ACCOUNT_NUMBER_FIELD","ACCOUNT_BALANCE_INFO","ACCOUNT_STATEMENT_URL","ACCOUNT_STATEMENT_EMAIL","ACCOUNT_ACTIVITY_INFO",
    "ACCOUNT_LOGIN_URL","RESET_PASSWORD_URL","RESET_PASSWORD_PHONE",
    "AUTOPAY_SETUP_URL","BILL_PAY_URL","TRANSFER_URL","WIRE_TRANSFER_INFO",
    "REFUND_STATUS_INFO","OTP_SUPPORT","PIN_CHANGE_URL","PIN_CHANGE_PHONE","PASSWORD_CHANGE_URL",
    "FEE_SCHEDULE_URL","ANNUAL_FEE_INFO","LATE_FEE_INFO","FOREIGN_TRANSACTION_FEE","INTEREST_RATE_INFO",
    "LOAN_SUPPORT_PHONE","LOAN_PAYMENT_URL","LOAN_PREPAYMENT_FEE","INSTALLMENT_INFO",
    "MOBILE_APP_APPSTORE_URL","MOBILE_APP_PLAYSTORE_URL",
    "ISSUER_WEBSITE","ISSUER_SUPPORT_PAGE"
]

# ----- Chuẩn hóa & đồn nghĩa -----
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def normalize_text(s: strg) -> str:
    if not isinstance(s, str): return ""
    t = s.strip().lower()
    t = strip_accents(t)
    t = re.sub(r"[\"'’`´]", " ", t)
    t = re.sub(r"[/|\\\-_,;:]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

VN_STOP = {"va","và","cua","của","la","là","cho","ve","về","tren","trên","duoi","dưới","tai","tại","theo",
           "voi","với","bao","gồm","bao gồm","hoac","hoặc","nhu","như","de","để","khach","hàng","khach hang",
           "nguoi","dung","người","dùng","so","số","ma","mã","thong","thông","tin","trong","cach","cách",
           "nhan","nhận","duoc","được","khi","neu","nếu","nhat","nhất","ho","họ","ten","tên"}
EN_STOP = {"the","a","an","of","for","on","to","in","by","with","and","or","at","from","is","are","be",
           "your","my","our","their","this","that","these","those","how","what","where","when",
           "customer","service","support"}  # (đã normalize, vẫn muốn giảm nhiễu)
STOP = VN_STOP | EN_STOP

SYNONYMS = [
    # kênh liên hệ
    (r"\b(cskh|customer\s*care|help\s*desk|helpdesk)\b", " support "),
    (r"\be[\- ]?mail(s)?\b", " email "),
    (r"\bmail\b", " email "),
    (r"\bhotline\b", " phone "),
    (r"\btelephone\b", " phone "),
    (r"\bphone(number| no)?\b", " phone "),
    (r"\bso\s+dien\s+thoai\b", " phone "),
    (r"\bdien\s+thoai\b", " phone "),
    (r"\bweb\s*site\b", " url "),
    (r"\bwebsite\b", " url "),
    (r"\burl\b", " url "),
    (r"\blink\b", " url "),
    (r"\blive\s*chat\b", " livechat "),
    (r"\bchat\s*live\b", " livechat "),
    (r"\blia\s*chi|dia\s*chi\b", " address "),
    (r"\bopening\s*hours|working\s*hours|gio\s*lam\s*viec|thoi\s*gian\s*lam\s*viec\b", " hours "),
    # thẻ & tài khoản
    (r"\bcard\s*number|so\s*the\b", " cardnumber "),
    (r"\bpin\b", " pin "),
    (r"\bcvv2?\b", " cvv "),
    (r"\bexpiry|expiration|ngay\s*het\s*han\b", " expiry "),
    (r"\bhan\s*muc|limit\b", " limit "),
    (r"\bnang\s*han\s*muc|increase\s*limit\b", " limitincrease "),
    (r"\bsao\s*ke|statement(s)?\b", " statement "),
    (r"\btai\s*khoan\b", " account "),
    (r"\bgiao\s*dich|transaction(s)?\b", " transaction "),
    (r"\bbank\s*login|dang\s*nhap|login\b", " login "),
    (r"\bpassword|mat\s*khau\b", " password "),
    (r"\breset|dat\s*lai\b", " reset "),
    (r"\bautopay|auto\s*payment\b", " autopay "),
    (r"\bbill\s*pay(ment)?\b", " billpay "),
    (r"\btransfer|chuyen\s*tien\b", " transfer "),
    (r"\bwire|swift|iban\b", " wire "),
    # phí, lãi
    (r"\bphi\s*thuong\s*nien|annual\s*fee\b", " annualfee "),
    (r"\bphi\b", " fee "),
    (r"\blai\s*suat|apr\b", " interest "),
    (r"\bngoai\s*te|foreign\b", " foreign "),
    (r"\brut\s*tien\s*mat|cash\s*advance\b", " cashadvance "),
    # bảo mật & rủi ro
    (r"\botp|2fa\b", " otp "),
    (r"\bfraud|lua\s*dao|phishing\b", " fraud "),
    (r"\bdispute|chargeback\b", " dispute "),
    (r"\blost|mat\b", " lost "),
    (r"\bstolen\b", " stolen "),
    (r"\bfreeze\b", " freeze "),
    (r"\bunfreeze\b", " unfreeze "),
    (r"\bblock|khoa\b", " block "),
    (r"\bunblock|mo\s*khoa\b", " unblock "),
    # ứng dụng
    (r"\bapp\s*store\b", " appstore "),
    (r"\bgoogle\s*play|play\s*store\b", " playstore "),
    # issuer
    (r"\bissuer\b", " issuer ")
]

def normalize_synonyms(s: str) -> str:
    t = normalize_text(s)
    for pat, rep in SYNONYMS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def token_set(s: str) -> set:
    toks = [w for w in re.split(r"\s+", s) if w]
    return {w for w in toks if w not in STOP}

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if (a or b) else 0.0

# ----- Anchor dictionary restricted to ALLOWED_KEYS -----
ANCHORS = [
    # key, display, tokens
    ("SUPPORT_PHONE", "Customer Support Phone Number", "support service customer phone hotline contact"),
    ("SUPPORT_EMAIL", "Customer Support Email Address", "support service customer email contact mail"),
    ("SUPPORT_WEBSITE", "Customer Support Website/Portal", "support service customer url website help center portal"),
    ("LIVE_CHAT_URL", "Live Chat URL", "livechat chat url website portal"),
    ("WORKING_HOURS", "Support Working Hours", "support service customer hours opening working"),
    ("BRANCH_ADDRESS", "Branch Address", "branch office address location dia chi"),
    ("BRANCH_LOCATOR_URL", "Branch Locator URL", "branch locator url website find branch"),
    ("ATM_LOCATOR_URL", "ATM Locator URL", "atm locator url website find atm"),

    ("LOST_STOLEN_CARD_PHONE", "Lost/Stolen Card Phone Number", "lost stolen card phone report"),
    ("CARD_ACTIVATION_PHONE", "Card Activation Phone Number", "activate activation card phone"),
    ("CARD_ACTIVATION_URL", "Card Activation Website", "activate activation card url website portal"),
    ("CARD_FREEZE_UNFREEZE", "Card Freeze/Unfreeze URL", "freeze unfreeze lock unlock card url"),
    ("MANAGE_CARDS_URL", "Manage Cards URL", "manage cards card url website portal"),

    ("CARD_NUMBER_FIELD", "Card Number", "cardnumber card number"),
    ("CARD_PIN_FIELD", "Card PIN", "pin card"),
    ("CARD_CVV_FIELD", "Card CVV", "cvv cvc"),
    ("CARD_EXPIRY_FIELD", "Card Expiry", "expiry expiration"),
    ("CREDIT_LIMIT_INFO", "Credit Limit", "limit credit han muc limitincrease"),
    ("CASH_ADVANCE_FEE", "Cash Advance Fee", "cashadvance fee rut tien mat"),

    ("ACCOUNT_NUMBER_FIELD", "Account Number", "account number so tai khoan"),
    ("ACCOUNT_BALANCE_INFO", "Account Balance", "account balance so du"),
    ("ACCOUNT_STATEMENT_URL", "Account Statement URL", "statement url website e-statement"),
    ("ACCOUNT_STATEMENT_EMAIL", "Account Statement Email", "statement email"),
    ("ACCOUNT_ACTIVITY_INFO", "Account Activity", "account activity giao dich transaction"),
    ("ACCOUNT_LOGIN_URL", "Online Banking Login URL", "login url website online banking"),
    ("RESET_PASSWORD_URL", "Reset Password URL", "reset password url"),
    ("RESET_PASSWORD_PHONE", "Reset Password Phone", "reset password phone"),

    ("AUTOPAY_SETUP_URL", "AutoPay Setup URL", "autopay automatic payment url"),
    ("BILL_PAY_URL", "Bill Payment URL", "billpay bill payment url"),
    ("TRANSFER_URL", "Transfer/Send Money URL", "transfer send money url"),
    ("WIRE_TRANSFER_INFO", "Wire Transfer Info", "wire transfer swift iban international"),

    ("REFUND_STATUS_INFO", "Refund Status Info", "refund status timeline"),
    ("OTP_SUPPORT", "OTP/2FA Support", "otp 2fa security"),
    ("PIN_CHANGE_URL", "Change PIN URL", "change pin url"),
    ("PIN_CHANGE_PHONE", "Change PIN Phone", "change pin phone"),
    ("PASSWORD_CHANGE_URL", "Change Password URL", "change password url"),

    ("FEE_SCHEDULE_URL", "Fee Schedule URL", "fee fees pricing schedule url"),
    ("ANNUAL_FEE_INFO", "Annual Fee", "annualfee phi thuong nien"),
    ("LATE_FEE_INFO", "Late Payment Fee", "late fee tra cham"),
    ("FOREIGN_TRANSACTION_FEE", "Foreign Transaction Fee", "foreign transaction fee"),
    ("INTEREST_RATE_INFO", "Interest Rate (APR)", "interest rate apr lai suat"),

    ("LOAN_SUPPORT_PHONE", "Loan Support Phone", "loan vay support phone"),
    ("LOAN_PAYMENT_URL", "Loan Payment URL", "loan payment url"),
    ("LOAN_PREPAYMENT_FEE", "Loan Prepayment Fee", "loan prepayment fee tra truoc"),
    ("INSTALLMENT_INFO", "Installment Plan Info", "installment emi tra gop"),

    ("MOBILE_APP_APPSTORE_URL", "iOS App Store URL", "appstore ios app url"),
    ("MOBILE_APP_PLAYSTORE_URL", "Android Play Store URL", "playstore android app url"),

    ("ISSUER_WEBSITE", "Issuer Website", "issuer url website"),
    ("ISSUER_SUPPORT_PAGE", "Issuer Support Page", "issuer support url website"),
]

# normalize anchors
ANCH = []
for key, display, toks in ANCHORS:
    if key not in ALLOWED_KEYS:  # safety guard
        continue
    norm = normalize_synonyms(toks)
    ANCH.append({
        "key": key,
        "display": display,
        "tokset": set(w for w in re.split(r"\s+", norm) if w and w not in STOP)
    })

def read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, dtype=str, encoding="utf-8")

def score_to_anchor(tokset: set, anchor_tokens: set) -> float:
    # dùng Jaccard đơn giản nhưng đã normalize + synonyms
    if not tokset and not anchor_tokens: return 1.0
    inter = len(tokset & anchor_tokens)
    union = len(tokset | anchor_tokens) or 1
    return inter / union

def auto_classify(label: str) -> Tuple[str, float]:
    """Trả về (anchor_key hoặc '', score)."""
    norm = normalize_synonyms(label)
    toks = token_set(norm)
    best_key, best_score = "", 0.0
    for a in ANCH:
        sc = score_to_anchor(toks, a["tokset"])
        if sc > best_score:
            best_key, best_score = a["key"], sc
    # Ngưỡng an toàn: 0.5 (vì đã có synonyms + tokens domain)
    return (best_key, best_score) if best_score >= 0.5 else ("", best_score)

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 1) Đọc mapping có sẵn
    base = read_csv(EXISTING_MAP)
    if "label" not in base.columns or "anchor_key" not in base.columns:
        raise SystemExit("EXISTING_MAP cần có cột 'label' và 'anchor_key'.")

    base = base[["label","anchor_key"]].copy()
    base["label"] = base["label"].astype(str).str.strip()
    base["anchor_key"] = base["anchor_key"].astype(str).str.strip()

    # Giữ lại chỉ các anchor_key thuộc ALLOWED_KEYS
    base_ok = base[base["anchor_key"].isin(ALLOWED_KEYS)].drop_duplicates(subset=["label"], keep="first")

    # 2) Tập nhãn bổ sung (nếu có)
    extra_labels = []
    if os.path.exists(EXTRA_LABELS):
        extra = read_csv(EXTRA_LABELS)
        if "label" in extra.columns:
            for _, r in extra.iterrows():
                lab = str(r["label"]).strip()
                if lab:
                    extra_labels.append(lab)

    # 3) Tạo tập labels cần xét:
    #    - Tất cả label đã có trong base (kể cả bị loại vì anchor_key không thuộc allowed)
    #    - + extra_labels (nếu có)
    all_labels = set(base["label"].astype(str).str.strip().tolist())
    all_labels.update(extra_labels)

    # 4) Ánh xạ cuối: ưu tiên base_ok, còn thiếu thì auto-classify trong ALLOWED_KEYS
    mapping = dict(zip(base_ok["label"], base_ok["anchor_key"]))

    conflicts = []  # khi base có anchor_key không thuộc allowed (bị bỏ), hoặc có nhãn trùng với auto khác key
    unmapped = []   # không đủ điểm auto

    # Index để tra cứu anchor_key gốc (kể cả không thuộc allowed) giúp log conflict minh bạch
    base_index = {str(r["label"]).strip(): str(r["anchor_key"]).strip() for _, r in base.iterrows()}

    for lab in sorted(all_labels):
        if not lab:
            continue
        if lab in mapping:
            continue  # đã có mapping hợp lệ từ base_ok
        # nếu base có anchor_key nhưng KHÔNG thuộc allowed -> ghi conflict, rồi thử auto map
        if lab in base_index and base_index[lab] not in ALLOWED_KEYS:
            old = base_index[lab]
        else:
            old = ""

        key, score = auto_classify(lab)
        if key:
            mapping[lab] = key
            if old and old != key:
                conflicts.append({"label": lab, "existing_anchor_key": old, "auto_anchor_key": key, "auto_score": round(score,4)})
        else:
            unmapped.append({"label": lab, "reason": "low_score_or_ambiguous", "max_score": round(score,4)})

    # 5) Xuất file
    out_final = os.path.join(OUT_DIR, "label_anchor_key_final.csv")
    out_conf  = os.path.join(OUT_DIR, "conflicts.csv")
    out_unmap = os.path.join(OUT_DIR, "unmapped.csv")

    final_df = pd.DataFrame(sorted(mapping.items(), key=lambda x: (x[0].lower(), x[1])), columns=["label","anchor_key"])
    final_df.to_csv(out_final, index=False, encoding="utf-8-sig", lineterminator="\n")

    pd.DataFrame(conflicts).to_csv(out_conf, index=False, encoding="utf-8-sig", lineterminator="\n")
    pd.DataFrame(unmapped).to_csv(out_unmap, index=False, encoding="utf-8-sig", lineterminator="\n")

    print("Đã ghi:", out_final)
    print("Đã ghi:", out_conf)
    print("Đã ghi:", out_unmap)

if __name__ == "__main__":
    main()