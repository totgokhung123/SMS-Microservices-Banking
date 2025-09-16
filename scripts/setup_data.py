import re, json, unicodedata
import pandas as pd
from pathlib import Path

# --- cấu hình ---
# CATEGORIES (UPPERCASE)
WHITELIST_CATEGORY = {
    "CARD","LOAN","TRANSFER","FEES","ACCOUNT","CONTACT","ATM","FIND","PASSWORD"
}
# INTENTS (lowercase)
WHITELIST_INTENT = {
    "activate_card","activate_card_international_usage",
    "apply_for_loan","apply_for_mortgage",
    "block_card","cancel_card","cancel_loan","cancel_mortgage","cancel_transfer",
    "check_card_annual_fee","check_current_balance_on_card","check_fees",
    "check_loan_payments","check_mortgage_payments","check_recent_transactions",
    "close_account","create_account","customer_service",
    "dispute_ATM_withdrawal","find_ATM","find_branch",
    "get_password","human_agent","make_transfer",
    "recover_swallowed_card","set_up_password"
}

# glossary tối thiểu (mở rộng dần)
GLOSSARY = {
    r"\bthu hút một thẻ\b": "kích hoạt thẻ",
    r"\bHiện hoạt thẻ\b": "kích hoạt thẻ",
    r"\bHiện (nhập|theo|tìm kiếm)\b": r"\1",
    r"\btham quan trang web\b": "truy cập trang web",
    r"\bTôi ở đây để giúp bạn\b": "Tôi có thể hỗ trợ Quý khách",
}

PLACEHOLDER_MAP = {
    r"\{\{\s*Credit Card\s*\}\}": "<CARD_PRODUCT>",
    r"\{\{\s*Customer Website URL\s*\}\}": "<ISSUER_WEBSITE>",
    r"\{\{\s*Customer Support Phone Number\s*\}\}": "<SUPPORT_PHONE>",
    r"\b(CVV|cvv)\b": "CVV"
}

PII_RISK_PATTERNS = [
    r"\b\d{16}\b",          # 16 chữ số liên tục (mạo hiểm là số thẻ)
    r"\b\d{3}\b\s*CVV",     # CVV pattern
    r"\bOTP\b", r"\bPIN\b"
]

def nfc(s): 
    return unicodedata.normalize("NFC", s)

def normalize_text(s: str) -> str:
    s = nfc(s.strip())
    s = re.sub(r"\s+", " ", s)
    # số thứ tự 1) -> 1.
    s = re.sub(r"(\d+)\)", r"\1.", s)
    # dấu cách trước dấu câu
    s = re.sub(r"\s+([,.!?;:])", r"\1", s)
    return s

def apply_glossary(s: str) -> str:
    for pat, rep in GLOSSARY.items():
        s = re.sub(pat, rep, s, flags=re.IGNORECASE)
    return s

def normalize_placeholders(s: str) -> str:
    for pat, rep in PLACEHOLDER_MAP.items():
        s = re.sub(pat, rep, s)
    # Trường hợp “<WORKING_HOURS>” tiếng Việt
    s = re.sub(r"\{\{\s*Customer Work Hours?\s*\}\}", "<WORKING_HOURS>", s, flags=re.I)
    return s

def fix_style_and_numbers(s: str) -> str:
    # Viết hoa đầu câu
    s = s[:1].upper() + s[1:] if s else s
    # Chuẩn danh sách đánh số: “1.” “2.” trên cùng một dòng -> thêm xuống dòng nếu cần
    s = re.sub(r"(\d\.)\s*", r"\1 ", s)
    return s

def add_pii_notice_if_needed(instr: str, resp: str) -> str:
    trigger = any(k in (instr+resp).lower() for k in ["kích hoạt thẻ","thẻ","pin","cvv","otp"])
    if trigger and "Không bao giờ cung cấp" not in resp:
        resp = resp.rstrip() + " Lưu ý: Quý khách KHÔNG bao giờ cung cấp số thẻ đầy đủ, CVV, PIN hoặc OTP qua chat. Khi cần xác minh, vui lòng liên hệ <SUPPORT_PHONE> hoặc <ISSUER_WEBSITE>."
    return resp

def quality_score(s: str) -> float:
    score = 1.0
    if "trường hợp của các trường hợp" in s: score -= 0.4
    if re.search(r"\bHiện\b\s", s): score -= 0.2
    if len(s) < 8: score -= 0.2
    return max(0.0, min(1.0, score))

def pii_risky(s: str) -> bool:
    return any(re.search(p, s) for p in PII_RISK_PATTERNS)

def clean_row(row):
    row["instruction"] = fix_style_and_numbers(
        apply_glossary(normalize_placeholders(normalize_text(row["instruction"])))
    )
    row["response"] = fix_style_and_numbers(
        apply_glossary(normalize_placeholders(normalize_text(row["response"])))
    )
    # giọng điệu
    row["response"] = row["response"].replace("Bạn", "Quý khách").replace("bạn", "Quý khách")
    # PII notice
    row["response"] = add_pii_notice_if_needed(row["instruction"], row["response"])
    # taxonomy
    cat = str(row["category"]).strip().lower()
    intent = str(row["intent"]).strip().lower()
    row["category"] = cat if cat in WHITELIST_CATEGORY else "other"
    row["intent"] = intent if intent in WHITELIST_INTENT else "other"
    # flags
    row["quality"] = min(quality_score(row["instruction"]), quality_score(row["response"]))
    row["pii_risky"] = pii_risky(row["response"])
    return row

# ví dụ sử dụng:
df = pd.read_csv("bank_vi_sample.csv", sep="\t")  # hoặc CSV chuẩn
df = df.fillna("")
df = df.apply(clean_row, axis=1)

# khử trùng lặp đơn giản
df["hash"] = (df["instruction"].str.lower() + "||" + df["response"].str.lower()).map(hash)
df = df.drop_duplicates(subset=["hash"]).drop(columns=["hash"])

# xuất JSONL cho Qwen
def to_chat_example(r):
    return {
      "messages": [
        {"role":"system","content":"Bạn là trợ lý tài chính ngân hàng. Không bao giờ yêu cầu CVV/PIN/OTP hoặc số thẻ đầy đủ. Luôn hướng dẫn khách qua kênh chính thức."},
        {"role":"user","content": r["instruction"]},
        {"role":"assistant","content": r["response"]}
      ],
      "meta": {"category": r["category"], "intent": r["intent"], "tags": r["tags"], "quality": r["quality"]}
    }

df_train = df.sample(frac=0.8, random_state=42)
df_temp = df.drop(df_train.index)
df_val = df_temp.sample(frac=0.5, random_state=42)
df_test = df_temp.drop(df_val.index)

for name, part in [("train", df_train), ("val", df_val), ("test", df_test)]:
    with open(f"qwen_vi_{name}.jsonl","w", encoding="utf-8") as f:
        for _, r in part.iterrows():
            f.write(json.dumps(to_chat_example(r), ensure_ascii=False)+"\n")
