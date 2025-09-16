# -*- coding: utf-8 -*-
"""
map_labels_to_taxonomy.py
-------------------------
Gom & gán ~2,300 labels vào taxonomy chi tiết theo "suy luận domain" (banking) mà không cần thư viện ngoài.
- Chuẩn hóa Việt–Anh, bỏ dấu, đồng nghĩa.
- Phân loại value_type (EMAIL/PHONE/URL/ADDRESS/HOURS/TEXT/NUMBER).
- So khớp tới "anchors" (danh mục mẫu chuẩn) với điểm số lai: token Jaccard + char-n-gram Dice + keyword-boost.
- Tách entity (brand/bank) nếu có (Visa/Mastercard/AmEx... + list bank VN phổ biến).
- Xuất 3 file: mapped_labels.csv, anchors_coverage.csv, unassigned_labels.csv

Cách dùng:
    CSV_INPUT=/path/labels_unique_edit.csv  CSV_OUTPUT_DIR=/path/out  python map_labels_to_taxonomy.py
"""
from __future__ import annotations
import os, re, unicodedata, json
from typing import List, Dict, Tuple
from collections import defaultdict, Counter
import pandas as pd

INPUT_CSV   = "/path/to/your/labels_unique_edit.csv"  # Thay bằng đường dẫn thực tế
OUTPUT_DIR  = "/path/to/your/output_directory"        # Thay bằng thư mục đầu ra
LABEL_COL   = "label"                                 # Tên cột chứa nhãn
COUNT_COL   = "count"                                 # Tên cột chứa số lượng

# -------------------- Helpers --------------------
def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    t = s.strip().lower()
    t = strip_accents(t)
    t = re.sub(r"[\"'’`´]", " ", t)
    t = re.sub(r"[/|\\\-_,;:]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def tokens(s: str) -> List[str]:
    return [w for w in re.split(r"\s+", s) if w]

VN_STOP = {"va","và","cua","của","la","là","cho","ve","về","tren","trên","duoi","dưới",
           "tai","tại","theo","voi","với","bao","gồm","bao gồm","hoac","hoặc","nhu","như",
           "de","để","khach","hàng","khach hang","nguoi","dung","người","dùng","so","số",
           "ma","mã","thong","thông","tin","trong","cach","cách","nhan","nhận","duoc","được",
           "khi","neu","nếu","va","và","nhat","nhất","ho","họ","ten","tên"}
EN_STOP = {"the","a","an","of","for","on","to","in","by","with","and","or","at","from",
           "is","are","be","your","my","our","their","customer","user","client","service",
           "services"}
STOP = VN_STOP | EN_STOP

SYNONYMS = [
    (r"\bsdt\b", "phone"),
    (r"\bso\s+dien\s+thoai\b", "phone"),
    (r"\bdien\s+thoai\b", "phone"),
    (r"\bhotline\b", "phone"),
    (r"\btelephone\b", "phone"),
    (r"\bphone(s|number| no)?\b", "phone"),
    (r"\bemail(s)?\b", "email"),
    (r"\bmail\b", "email"),
    (r"\bweb\s*site\b", "url"),
    (r"\bwebsite\b", "url"),
    (r"\burl\b", "url"),
    (r"\blink\b", "url"),
    (r"\bapp\s*store\b", "appstore"),
    (r"\bgoogle\s*play\b", "playstore"),
    (r"\bplay\s*store\b", "playstore"),
    (r"\blia\s+chi\b", "address"),
    (r"\bdia\s+chi\b", "address"),
    (r"\baddress(es)?\b", "address"),
    (r"\bgio\s+lam\s+viec\b", "hours"),
    (r"\bthoi\s+gian\s+lam\s+viec\b", "hours"),
    (r"\bopening\s*hours\b", "hours"),
    (r"\bworking\s*hours\b", "hours"),
    (r"\bcard\s*number\b", "cardnumber"),
    (r"\bso\s+the\b", "cardnumber"),
    (r"\bthe\b", "card"),
    (r"\bpin\b", "pin"),
    (r"\botp\b", "otp"),
    (r"\b2fa\b", "2fa"),
    (r"\bcvv\b", "cvv"),
    (r"\bcvv2\b", "cvv"),
    (r"\bexpiry\b", "expiry"),
    (r"\bexpiration\b", "expiry"),
    (r"\bngay\s+het\s+han\b", "expiry"),
    (r"\bfee(s)?\b", "fee"),
    (r"\bphi\b", "fee"),
    (r"\binterest\b", "interest"),
    (r"\blai\s*suat\b", "interest"),
    (r"\blimit\b", "limit"),
    (r"\bhan\s+muc\b", "limit"),
    (r"\btransaction(s)?\b", "transaction"),
    (r"\bgiao\s+dich\b", "transaction"),
    (r"\bstatement(s)?\b", "statement"),
    (r"\bbank\s*statement\b", "statement"),
    (r"\bextracto\b", "statement"),
    (r"\baccount\b", "account"),
    (r"\btai\s*khoan\b", "account"),
    (r"\bissuer\b", "issuer"),
    (r"\bbranch(es)?\b", "branch"),
    (r"\batm(s)?\b", "atm"),
    (r"\blocator\b", "locator"),
    (r"\bdispute(s)?\b", "dispute"),
    (r"\bchargeback\b", "dispute"),
    (r"\bfraud\b", "fraud"),
    (r"\bphishing\b", "phishing"),
    (r"\bregister\b", "register"),
    (r"\bactivate\b", "activate"),
    (r"\bkich\s*hoat\b", "activate"),
    (r"\bun\s*block\b", "unblock"),
    (r"\blo(s|st)\b", "lost"),
    (r"\bstolen\b", "stolen"),
    (r"\breplace(ment)?\b", "replace"),
    (r"\bbao\s*mat\b", "security"),
    (r"\bsecurity\b", "security"),
    (r"\blogin\b", "login"),
    (r"\bdang\s*nhap\b", "login"),
    (r"\bpassword\b", "password"),
    (r"\bmat\s*khau\b", "password"),
    (r"\bres(et|end)\b", "reset"),
    (r"\bdoi\b", "change"),
    (r"\bchange\b", "change")
]

BANK_BRANDS = {
    # Card schemes
    "visa","mastercard","american express","amex","discover","jcb","unionpay",
    # VN banks (phổ biến)
    "vietcombank","vcb","techcombank","tcb","bidv","vietinbank","ctg","agribank","mbbank","mb",
    "tpbank","vpbank","acb","sacombank","shb","seabank","hdbank","ocb","vib","pgbank","abbank",
    "eximbank","eib","scb","ncb","vrb","pvcombank","bac a bank","baoviet bank","kienlongbank",
    "saigonbank","namabank","lpbank","lpb","msb"
}

def normalize_synonyms(s: str) -> str:
    t = normalize_text(s)
    for pat, rep in SYNONYMS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def token_set(s: str) -> set:
    toks = [w for w in re.split(r"\s+", s) if w]
    return set(w for w in toks if w not in STOP)

def char_ngrams(s: str, n=3) -> set:
    s2 = re.sub(r"\s+", " ", s)
    if len(s2) < n: return {s2} if s2 else set()
    return {s2[i:i+n] for i in range(len(s2)-n+1)}

def dice(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return 2*len(a & b) / (len(a) + len(b)) if (a or b) else 0.0

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / len(a | b) if (a or b) else 0.0

def detect_value_type(text_norm: str) -> str:
    t = " " + text_norm + " "
    if re.search(r"\bemail\b", t): return "EMAIL"
    if re.search(r"\bphone\b", t): return "PHONE"
    if re.search(r"\burl\b|\blink\b|\bwebsite\b", t): return "URL"
    if re.search(r"\baddress\b", t): return "ADDRESS"
    if re.search(r"\bhours\b", t): return "HOURS"
    # textual fields that look numeric-ish
    if re.search(r"\b(cardnumber|pin|cvv|otp)\b", t): return "NUMBER"
    return "TEXT"

def extract_entity(text_norm: str) -> str:
    # tìm brand/bank
    for name in sorted(BANK_BRANDS, key=lambda x: -len(x)):
        if f" {name} " in f" {text_norm} ":
            return name.upper().replace(" ", "_")
    return ""

# -------------------- Taxonomy Anchors --------------------
# Mỗi anchor gồm: key, display, tokens (string), category, value_type (EMAIL/PHONE/URL/ADDRESS/HOURS/TEXT/NUMBER)
ANCHORS = [
    # Contact & Support
    ("SUPPORT_PHONE", "Customer Support Phone Number", "support phone hotline contact", "contact", "PHONE"),
    ("SUPPORT_EMAIL", "Customer Support Email Address", "support email contact mail", "contact", "EMAIL"),
    ("SUPPORT_WEBSITE", "Customer Support Website/Portal", "support url website help center portal", "contact", "URL"),
    ("LIVE_CHAT_URL", "Live Chat URL", "chat live url website portal", "contact", "URL"),
    ("WORKING_HOURS", "Support Working Hours", "support hours working opening", "contact", "HOURS"),
    ("BRANCH_ADDRESS", "Branch Address", "branch address office location", "contact", "ADDRESS"),
    ("BRANCH_LOCATOR_URL", "Branch Locator URL", "branch locator url website find branch", "contact", "URL"),
    ("ATM_LOCATOR_URL", "ATM Locator URL", "atm locator url website find atm", "contact", "URL"),
    # Card emergencies / activation
    ("LOST_STOLEN_CARD_PHONE", "Lost/Stolen Card Phone Number", "lost stolen card phone report", "card", "PHONE"),
    ("CARD_ACTIVATION_PHONE", "Card Activation Phone Number", "activate activation card phone", "card", "PHONE"),
    ("CARD_ACTIVATION_URL", "Card Activation Website", "activate activation card url website portal", "card", "URL"),
    ("CARD_REPLACEMENT_PHONE", "Card Replacement Phone", "replace replacement card phone damaged", "card", "PHONE"),
    ("CARD_FREEZE_UNFREEZE", "Card Freeze/Unfreeze URL", "freeze unfreeze lock unlock card url", "card", "URL"),
    ("MANAGE_CARDS_URL", "Manage Cards URL", "manage cards url website portal", "card", "URL"),
    # Card details
    ("CARD_NUMBER_FIELD", "Card Number", "cardnumber card number", "card", "NUMBER"),
    ("CARD_PIN_FIELD", "Card PIN", "pin card", "card", "NUMBER"),
    ("CARD_CVV_FIELD", "Card CVV", "cvv cvc", "card", "NUMBER"),
    ("CARD_EXPIRY_FIELD", "Card Expiry", "expiry expiration", "card", "TEXT"),
    ("CREDIT_LIMIT_INFO", "Credit Limit", "limit credit han muc", "card", "TEXT"),
    ("CREDIT_LIMIT_INCREASE_URL", "Credit Limit Increase URL", "limit increase url request", "card", "URL"),
    # Account
    ("ACCOUNT_NUMBER_FIELD", "Account Number", "account number so tai khoan", "account", "NUMBER"),
    ("ACCOUNT_BALANCE_INFO", "Account Balance", "account balance so du", "account", "TEXT"),
    ("ACCOUNT_STATEMENT_URL", "Account Statement URL", "statement url website", "account", "URL"),
    ("ACCOUNT_STATEMENT_EMAIL", "Account Statement Email", "statement email", "account", "EMAIL"),
    ("ACCOUNT_ACTIVITY_INFO", "Account Activity", "account activity giao dich", "account", "TEXT"),
    ("ACCOUNT_LOGIN_URL", "Online Banking Login URL", "login url website online banking", "account", "URL"),
    ("RESET_PASSWORD_URL", "Reset Password URL", "reset password url", "account", "URL"),
    ("RESET_PASSWORD_PHONE", "Reset Password Phone", "reset password phone", "account", "PHONE"),
    # Payments & transfers
    ("AUTOPAY_SETUP_URL", "AutoPay Setup URL", "autopay automatic payment url", "payments", "URL"),
    ("TRANSFER_URL", "Transfer/Send Money URL", "transfer send money url", "payments", "URL"),
    ("WIRE_TRANSFER_INFO", "Wire Transfer Info", "wire transfer swift iban", "payments", "TEXT"),
    ("INTERNATIONAL_TRANSFER_INFO", "International Transfer Info", "international transfer swift iban", "payments", "TEXT"),
    # Disputes & fraud
    ("DISPUTE_PHONE", "Transaction Dispute Phone", "dispute chargeback phone", "disputes", "PHONE"),
    ("DISPUTE_URL", "Transaction Dispute URL", "dispute chargeback url", "disputes", "URL"),
    ("FRAUD_REPORT_PHONE", "Fraud Report Phone", "fraud scam phishing report phone", "security", "PHONE"),
    ("FRAUD_REPORT_URL", "Fraud Report URL", "fraud scam phishing report url", "security", "URL"),
    # Fees & rates
    ("FEE_SCHEDULE_URL", "Fee Schedule URL", "fee fees pricing schedule url", "fees", "URL"),
    ("ANNUAL_FEE_INFO", "Annual Fee", "annual fee phi thuong nien", "fees", "TEXT"),
    ("LATE_FEE_INFO", "Late Payment Fee", "late fee tra cham", "fees", "TEXT"),
    ("FOREIGN_TRANSACTION_FEE", "Foreign Transaction Fee", "foreign transaction fee", "fees", "TEXT"),
    ("INTEREST_RATE_INFO", "Interest Rate (APR)", "interest rate apr lai suat", "fees", "TEXT"),
    # Loans
    ("LOAN_SUPPORT_PHONE", "Loan Support Phone", "loan vay support phone", "loans", "PHONE"),
    ("LOAN_PAYMENT_URL", "Loan Payment URL", "loan payment url", "loans", "URL"),
    ("LOAN_PREPAYMENT_FEE", "Loan Prepayment Fee", "loan prepayment fee tra truoc", "loans", "TEXT"),
    # Security / auth
    ("OTP_SUPPORT", "OTP/2FA Support", "otp 2fa security", "security", "TEXT"),
    ("PIN_CHANGE_URL", "Change PIN URL", "change pin url", "security", "URL"),
    ("PIN_CHANGE_PHONE", "Change PIN Phone", "change pin phone", "security", "PHONE"),
    ("PASSWORD_CHANGE_URL", "Change Password URL", "change password url", "security", "URL"),
    # Apps & status
    ("MOBILE_APP_APPSTORE_URL", "iOS App Store URL", "appstore ios app url", "apps", "URL"),
    ("MOBILE_APP_PLAYSTORE_URL", "Android Play Store URL", "playstore android app url", "apps", "URL"),
    ("SYSTEM_STATUS_URL", "System Status URL", "system status outage url", "apps", "URL"),
    # Issuer/bank sites
    ("ISSUER_WEBSITE", "Issuer Website", "issuer url website", "issuer", "URL"),
    ("ISSUER_SUPPORT_PAGE", "Issuer Support Page", "issuer support url website", "issuer", "URL"),
]

def anchor_to_struct():
    L = []
    for key, display, toks, cat, vtype in ANCHORS:
        norm = normalize_synonyms(toks)
        L.append({
            "key": key,
            "display": display,
            "category": cat,
            "value_type": vtype,
            "norm": norm,
            "tokset": token_set(norm),
            "ng3": char_ngrams(norm, 3),
            "ng4": char_ngrams(norm, 4)
        })
    return L

ANCH = anchor_to_struct()

def keyword_boost(label_norm: str, anchor: dict) -> float:
    # tăng điểm nếu một số từ khóa trùng nhau quan trọng xuất hiện
    t = f" {label_norm} "
    boost = 0.0
    key_tokens = {"lost","stolen","activate","dispute","fraud","fee","interest","limit",
                  "branch","atm","locator","reset","password","pin","otp","login","statement"}
    for k in key_tokens & anchor["tokset"]:
        if f" {k} " in t:
            boost += 0.05  # mỗi từ khóa +0.05, tổng có trần
    return min(boost, 0.25)

def score_label_to_anchor(label_text: str, label_norm: str, anchor: dict) -> float:
    # tính điểm lai
    ltoks = token_set(label_norm)
    lng3  = char_ngrams(label_norm, 3)
    lng4  = char_ngrams(label_norm, 4)

    t_j = jaccard(ltoks, anchor["tokset"])    # token jaccard
    c_d = 0.6*dice(lng3, anchor["ng3"]) + 0.4*dice(lng4, anchor["ng4"])  # char dice
    b   = keyword_boost(label_norm, anchor)

    # trọng số: token 0.5, char 0.4, boost 0.1 (b capped by 0.25)
    return 0.5*t_j + 0.4*c_d + 0.1*b

def main():
    if not INPUT_CSV:
        raise SystemExit("Thiếu CSV_INPUT (labels CSV).")
    if not OUTPUT_DIR:
        OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(INPUT_CSV)), "mapped_out")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    try:
        df = pd.read_csv(INPUT_CSV, dtype=str, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_CSV, dtype=str, encoding="utf-8")

    if LABEL_COL not in df.columns:
        # đoán cột
        for c in df.columns:
            if c.lower().strip() == "label":
                global LABEL_COL
                LABEL_COL = c
                break
        else:
            raise SystemExit(f"Không tìm thấy cột '{LABEL_COL}'. Các cột: {list(df.columns)}")

    if COUNT_COL not in df.columns:
        df[COUNT_COL] = 1

    rows = []
    cov_counter = Counter()
    unassigned = []

    for _, r in df.iterrows():
        label = str(r[LABEL_COL]).strip()
        try:
            cnt = int(str(r[COUNT_COL]))
        except:
            cnt = 1

        norm1 = normalize_synonyms(label)
        vtype = detect_value_type(norm1)
        entity = extract_entity(norm1)

        # lọc anchors theo value_type (nếu TEXT thì không lọc để không bỏ lỡ)
        candidates = [a for a in ANCH if (vtype == "TEXT" or a["value_type"] == vtype)]
        # chấm điểm
        best_key, best_display, best_cat, best_vtype, best_score = "", "", "", "", 0.0
        for a in candidates:
            sc = score_label_to_anchor(label, norm1, a)
            if sc > best_score:
                best_key, best_display, best_cat, best_vtype, best_score = a["key"], a["display"], a["category"], a["value_type"], sc

        # ngưỡng quyết định: tùy kiểu
        # PHONE/EMAIL/URL/ADDRESS/HOURS → dễ nhận dạng → ngưỡng 0.52; NUMBER → 0.50; TEXT → 0.58
        thr = 0.58 if vtype == "TEXT" else (0.50 if vtype == "NUMBER" else 0.52)

        if best_key and best_score >= thr:
            rows.append({
                "label": label,
                "count": cnt,
                "entity": entity,
                "value_type": vtype,
                "anchor_key": best_key,
                "anchor_display": best_display,
                "anchor_category": best_cat,
                "score": round(best_score, 4),
                "normalized": norm1
            })
            cov_counter[(best_key, best_display, best_cat, best_vtype)] += cnt
        else:
            unassigned.append({
                "label": label,
                "count": cnt,
                "entity": entity,
                "value_type": vtype,
                "normalized": norm1,
                "note": "below_threshold"
            })

    mapped_df = pd.DataFrame(rows).sort_values(["anchor_category","anchor_key","score","count"], ascending=[True,True,False,False])
    un_df     = pd.DataFrame(unassigned).sort_values(["value_type","count"], ascending=[True,False])

    cov_rows = []
    for (k, d, c, vt), s in sorted(cov_counter.items(), key=lambda x: (-x[1], x[0])):
        cov_rows.append({"anchor_key": k, "anchor_display": d, "anchor_category": c, "value_type": vt, "total_count": s})
    cov_df = pd.DataFrame(cov_rows)

    out1 = os.path.join(OUTPUT_DIR, "mapped_labels.csv")
    out2 = os.path.join(OUTPUT_DIR, "anchors_coverage.csv")
    out3 = os.path.join(OUTPUT_DIR, "unassigned_labels.csv")

    mapped_df.to_csv(out1, index=False, encoding="utf-8-sig", lineterminator="\n")
    cov_df.to_csv(out2, index=False, encoding="utf-8-sig", lineterminator="\n")
    un_df.to_csv(out3, index=False, encoding="utf-8-sig", lineterminator="\n")

    print("WROTE:", out1)
    print("WROTE:", out2)
    print("WROTE:", out3)

if __name__ == "__main__":
    main()
    
    
    