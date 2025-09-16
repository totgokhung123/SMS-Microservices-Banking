# Advanced auto-classification for unassigned labels:
# - Clusters unassigned labels by token Jaccard
# - Maps clusters to existing anchors (if strong match)
# - Proposes NEW anchors for clusters that don't match
# - Writes several CSVs for you to review/apply

import os, re, unicodedata, math
from pathlib import Path
from collections import defaultdict, Counter
import pandas as pd

BASE_INPUT = "/mnt/data/unassigned_labels.csv"
OUTDIR = Path("/mnt/data/unassigned_autoclass")
OUTDIR.mkdir(parents=True, exist_ok=True)

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFD", s)
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")

# Stopwords (VN + EN) kept minimal to avoid over-removal
VN_STOP = {
    "va","và","cua","của","la","là","cho","ve","về","tren","trên","duoi","dưới","tai","tại","theo",
    "voi","với","bao","gồm","bao gồm","hoac","hoặc","nhu","như","de","để","khach","hàng","khach hang",
    "nguoi","dung","người","dùng","so","số","ma","mã","thong","thông","tin","trong","cach","cách",
    "nhan","nhận","duoc","được","khi","neu","nếu","nhat","nhất","ho","họ","ten","tên","thanh","cua"
}
EN_STOP = {
    "the","a","an","of","for","on","to","in","by","with","and","or","at","from","is","are","be",
    "your","my","our","their","this","that","these","those","how","what","where","when"
}
STOP = VN_STOP | EN_STOP

# Synonyms: normalize VN-EN to common tokens
SYNONYMS = [
    # contact/support
    (r"\bcskh\b", " support "),
    (r"\bcustomer\s*care\b", " support "),
    (r"\bcare\b", " support "),
    (r"\bhelp\s*desk\b", " support "),
    (r"\bhelpdesk\b", " support "),
    (r"\bdich\s*vu\b", " service "),
    (r"\bcham\s*soc\s*khach\s*hang\b", " support "),
    (r"\bkhach\s*hang\b", " customer "),
    (r"\bho\s*tro\b", " support "),
    # channels
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
    (r"\bchat\s*live\b", " livechat "),
    (r"\blive\s*chat\b", " livechat "),
    # address/hours
    (r"\blia\s*chi\b", " address "),
    (r"\bdia\s*chi\b", " address "),
    (r"\bopening\s*hours\b", " hours "),
    (r"\bworking\s*hours\b", " hours "),
    (r"\bgio\s*lam\s*viec\b", " hours "),
    (r"\bthoi\s*gian\s*lam\s*viec\b", " hours "),
    # card/account specifics
    (r"\bcard\s*number\b", " cardnumber "),
    (r"\bso\s+the\b", " cardnumber "),
    (r"\bthe\b", " card "),
    (r"\btai\s*khoan\b", " account "),
    (r"\bsao\s*ke\b", " statement "),
    (r"\bstatement(s)?\b", " statement "),
    (r"\brut\s*tien\s*mat\b", " cashadvance "),
    (r"\bcash\s*advance\b", " cashadvance "),
    (r"\bhan\s*muc\b", " limit "),
    (r"\blimit\b", " limit "),
    (r"\bnang\s*han\s*muc\b", " limitincrease "),
    (r"\bincrease\s*limit\b", " limitincrease "),
    (r"\blai\s*suat\b", " interest "),
    (r"\bapr\b", " interest "),
    (r"\bphi\s*thuong\s*nien\b", " annualfee "),
    (r"\bannual\s*fee\b", " annualfee "),
    (r"\bphi\b", " fee "),
    (r"\btransaction(s)?\b", " transaction "),
    (r"\bgiao\s*dich\b", " transaction "),
    (r"\bdispute(s)?\b", " dispute "),
    (r"\bchargeback\b", " dispute "),
    (r"\bfraud\b", " fraud "),
    (r"\bphishing\b", " phishing "),
    (r"\bactivate|kich\s*hoat\b", " activate "),
    (r"\bblock|khoa\b", " block "),
    (r"\bunblock|mo\s*khoa\b", " unblock "),
    (r"\bfreeze\b", " freeze "),
    (r"\bunfreeze\b", " unfreeze "),
    (r"\blost|mat\b", " lost "),
    (r"\bstolen\b", " stolen "),
    (r"\breplace(ment)?\b", " replace "),
    (r"\blogin|dang\s*nhap\b", " login "),
    (r"\bpassword|mat\s*khau\b", " password "),
    (r"\bres(et|end)\b", " reset "),
    (r"\bpin\b", " pin "),
    (r"\bcvv2?\b", " cvv "),
    (r"\bexpiry|expiration|ngay\s*het\s*han\b", " expiry "),
    (r"\botp\b", " otp "),
    (r"\b2fa\b", " 2fa "),
    (r"\bwire\b", " wire "),
    (r"\bib an\b", " iban "),
    (r"\bswift\b", " swift "),
    (r"\bkyc|din h\s*danh|xac\s*thuc\b", " kyc "),
    (r"\bhoan\s*tien|cash\s*back|cashback\b", " cashback "),
    (r"\bdiem\s*thuong|rewards|points?\b", " rewards "),
    (r"\btra\s*gop|installment|emi\b", " installment "),
    (r"\bpromotion|khuyen\s*mai|uu\s*dai\b", " promo "),
    (r"\bmerchant\b", " merchant "),
    (r"\bmcc\b", " mcc "),
]

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    t = s.strip().lower()
    t = strip_accents(t)
    t = re.sub(r"[\"'’`´]", " ", t)
    t = re.sub(r"[/|\\\-_,;:]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

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

# Existing anchors (broader taxonomy), kept in sync with previous script, plus some extra
ANCHORS = [
    ("SUPPORT_PHONE","Customer Support Phone Number","support phone hotline contact","contact","PHONE"),
    ("SUPPORT_EMAIL","Customer Support Email Address","support email contact mail","contact","EMAIL"),
    ("SUPPORT_WEBSITE","Customer Support Website/Portal","support url website help center portal","contact","URL"),
    ("LIVE_CHAT_URL","Live Chat URL","livechat url website portal","contact","URL"),
    ("WORKING_HOURS","Support Working Hours","support hours working opening","contact","HOURS"),
    ("BRANCH_ADDRESS","Branch Address","branch address office location","contact","ADDRESS"),
    ("BRANCH_LOCATOR_URL","Branch Locator URL","branch locator url website find branch","contact","URL"),
    ("ATM_LOCATOR_URL","ATM Locator URL","atm locator url website find atm","contact","URL"),

    ("LOST_STOLEN_CARD_PHONE","Lost/Stolen Card Phone Number","lost stolen card phone report","card","PHONE"),
    ("CARD_ACTIVATION_PHONE","Card Activation Phone Number","activate activation card phone","card","PHONE"),
    ("CARD_ACTIVATION_URL","Card Activation Website","activate activation card url website portal","card","URL"),
    ("CARD_REPLACEMENT_PHONE","Card Replacement Phone","replace replacement card phone damaged","card","PHONE"),
    ("CARD_FREEZE_UNFREEZE","Card Freeze/Unfreeze URL","freeze unfreeze lock unlock card url","card","URL"),
    ("MANAGE_CARDS_URL","Manage Cards URL","manage cards url website portal","card","URL"),
    ("CARD_DELIVERY_STATUS_URL","Card Delivery/Tracking URL","card delivery shipping tracking url","card","URL"),

    ("CARD_NUMBER_FIELD","Card Number","cardnumber card number","card","NUMBER"),
    ("CARD_PIN_FIELD","Card PIN","pin card","card","NUMBER"),
    ("CARD_CVV_FIELD","Card CVV","cvv cvc","card","NUMBER"),
    ("CARD_EXPIRY_FIELD","Card Expiry","expiry expiration","card","TEXT"),
    ("CREDIT_LIMIT_INFO","Credit Limit","limit credit han muc","card","TEXT"),
    ("CREDIT_LIMIT_INCREASE_URL","Credit Limit Increase URL","limitincrease request url","card","URL"),
    ("CASH_ADVANCE_FEE","Cash Advance Fee","cashadvance fee rut tien mat","fees","TEXT"),

    ("ACCOUNT_NUMBER_FIELD","Account Number","account number so tai khoan","account","NUMBER"),
    ("ACCOUNT_BALANCE_INFO","Account Balance","account balance so du","account","TEXT"),
    ("ACCOUNT_STATEMENT_URL","Account Statement URL","statement url website e-statement","account","URL"),
    ("ACCOUNT_STATEMENT_EMAIL","Account Statement Email","statement email","account","EMAIL"),
    ("ACCOUNT_ACTIVITY_INFO","Account Activity","account activity giao dich","account","TEXT"),
    ("ACCOUNT_LOGIN_URL","Online Banking Login URL","login url website online banking","account","URL"),
    ("RESET_PASSWORD_URL","Reset Password URL","reset password url","account","URL"),
    ("RESET_PASSWORD_PHONE","Reset Password Phone","reset password phone","account","PHONE"),

    ("AUTOPAY_SETUP_URL","AutoPay Setup URL","autopay automatic payment url","payments","URL"),
    ("BILL_PAY_URL","Bill Payment URL","bill pay payment url","payments","URL"),
    ("TRANSFER_URL","Transfer/Send Money URL","transfer send money url","payments","URL"),
    ("WIRE_TRANSFER_INFO","Wire Transfer Info","wire transfer swift iban international","payments","TEXT"),

    ("DISPUTE_PHONE","Transaction Dispute Phone","dispute chargeback phone","disputes","PHONE"),
    ("DISPUTE_URL","Transaction Dispute URL","dispute chargeback url","disputes","URL"),
    ("REFUND_STATUS_INFO","Refund Status Info","refund status timeline","disputes","TEXT"),

    ("FRAUD_REPORT_PHONE","Fraud Report Phone","fraud scam phishing report phone","security","PHONE"),
    ("FRAUD_REPORT_URL","Fraud Report URL","fraud scam phishing report url","security","URL"),
    ("OTP_SUPPORT","OTP/2FA Support","otp 2fa security","security","TEXT"),
    ("PIN_CHANGE_URL","Change PIN URL","change pin url","security","URL"),
    ("PIN_CHANGE_PHONE","Change PIN Phone","change pin phone","security","PHONE"),
    ("PASSWORD_CHANGE_URL","Change Password URL","change password url","security","URL"),

    ("FEE_SCHEDULE_URL","Fee Schedule URL","fee fees pricing schedule url","fees","URL"),
    ("ANNUAL_FEE_INFO","Annual Fee","annualfee phi thuong nien","fees","TEXT"),
    ("LATE_FEE_INFO","Late Payment Fee","late fee tra cham","fees","TEXT"),
    ("FOREIGN_TRANSACTION_FEE","Foreign Transaction Fee","foreign transaction fee","fees","TEXT"),
    ("INTEREST_RATE_INFO","Interest Rate (APR)","interest rate apr lai suat","fees","TEXT"),

    ("LOAN_SUPPORT_PHONE","Loan Support Phone","loan vay support phone","loans","PHONE"),
    ("LOAN_PAYMENT_URL","Loan Payment URL","loan payment url","loans","URL"),
    ("LOAN_PREPAYMENT_FEE","Loan Prepayment Fee","loan prepayment fee tra truoc","loans","TEXT"),
    ("INSTALLMENT_INFO","Installment Plan Info","installment emi tra gop","loans","TEXT"),

    ("REWARDS_INFO","Rewards/Points Info","rewards points diem thuong","rewards","TEXT"),
    ("REWARDS_REDEMPTION_URL","Rewards Redemption URL","rewards redemption redeem url","rewards","URL"),
    ("CASHBACK_INFO","Cashback Info","cashback hoan tien","rewards","TEXT"),
    ("PROMO_OFFERS_URL","Promotions/Offers URL","promo khuyen mai uu dai url","rewards","URL"),

    ("MOBILE_APP_APPSTORE_URL","iOS App Store URL","appstore ios app url","apps","URL"),
    ("MOBILE_APP_PLAYSTORE_URL","Android Play Store URL","playstore android app url","apps","URL"),
    ("SYSTEM_STATUS_URL","System Status URL","system status outage url","apps","URL"),

    ("ISSUER_WEBSITE","Issuer Website","issuer url website","issuer","URL"),
    ("ISSUER_SUPPORT_PAGE","Issuer Support Page","issuer support url website","issuer","URL"),
]

def anchor_struct():
    L = []
    for key, display, toks, cat, vtype in ANCHORS:
        norm = normalize_synonyms(toks)
        L.append({
            "key": key, "display": display, "category": cat, "value_type": vtype,
            "tokset": token_set(norm)
        })
    return L
ANCH = anchor_struct()

def value_type_guess(norm: str) -> str:
    t = f" {norm} "
    if " email " in t: return "EMAIL"
    if " phone " in t: return "PHONE"
    if " url " in t: return "URL"
    if " address " in t: return "ADDRESS"
    if " hours " in t: return "HOURS"
    if any(x in t for x in [" pin "," cvv "," otp "," cardnumber "]): return "NUMBER"
    return "TEXT"

def category_guess(tok: set) -> str:
    # heuristic by presence of tokens
    tests = [
        ("card", "card"),
        ("account", "account"),
        ("loan", "loans"),
        ("installment", "loans"),
        ("wire", "payments"),
        ("transfer", "payments"),
        ("bill", "payments"),
        ("autopay", "payments"),
        ("dispute", "disputes"),
        ("chargeback", "disputes"),
        ("fraud", "security"),
        ("phishing", "security"),
        ("password", "security"),
        ("pin", "security"),
        ("otp", "security"),
        ("fee", "fees"),
        ("interest", "fees"),
        ("annualfee", "fees"),
        ("promo", "rewards"),
        ("rewards", "rewards"),
        ("cashback", "rewards"),
        ("appstore", "apps"),
        ("playstore", "apps"),
        ("issuer", "issuer"),
        ("branch", "contact"),
        ("atm", "contact"),
        ("support", "contact"),
        ("service", "contact"),
        ("customer", "contact"),
    ]
    for k, cat in tests:
        if k in tok:
            return cat
    return "other"

def to_upper_snake(s: str) -> str:
    s0 = strip_accents(s).lower()
    s0 = re.sub(r"[^a-z0-9]+", "_", s0)
    s0 = re.sub(r"_+", "_", s0).strip("_")
    if not s0:
        s0 = "PLACEHOLDER"
    return s0.upper()

# Load input
try:
    df = pd.read_csv(BASE_INPUT, dtype=str, encoding="utf-8-sig")
except UnicodeDecodeError:
    df = pd.read_csv(BASE_INPUT, dtype=str, encoding="utf-8")

if "label" not in df.columns:
    raise SystemExit("unassigned_labels.csv cần có cột 'label'.")

if "count" not in df.columns:
    df["count"] = 1

# Prepare normalized + tokens
items = []
for _, r in df.iterrows():
    label = str(r["label"]).strip()
    cnt = int(str(r["count"])) if str(r["count"]).isdigit() else 1
    norm = normalize_synonyms(label)
    toks = token_set(norm)
    vt = value_type_guess(norm)
    items.append({"label": label, "count": cnt, "norm": norm, "toks": toks, "vt": vt})

# Greedy clustering by Jaccard
# Sort by frequency desc to seed with common labels
items_sorted = sorted(items, key=lambda x: (-x["count"], x["label"]))
visited = set()
clusters = []  # list of dict: {"seed_idx": i, "members": [indices], "union": set()}

JACCARD_THR = 0.65  # moderate threshold, we will still verify per-cluster mapping later

for i, it in enumerate(items_sorted):
    if i in visited:
        continue
    # start a new cluster
    cluster_members = [i]
    visited.add(i)
    base = it["toks"]
    union = set(base)
    # compare to others
    for j in range(i+1, len(items_sorted)):
        if j in visited:
            continue
        jt = items_sorted[j]["toks"]
        if not jt and not base:
            sim = 1.0
        else:
            sim = jaccard(union, jt)
        if sim >= JACCARD_THR:
            cluster_members.append(j)
            visited.add(j)
            union |= jt
    clusters.append({"seed": i, "members": cluster_members, "union": union})

# Map clusters to existing anchors when possible; otherwise propose new anchor
mapped_rows = []
new_anchor_rows = []
still_rows = []

def score_tok_to_anchor(tokset, anchor_toks):
    if not tokset and not anchor_toks:
        return 1.0
    return jaccard(tokset, anchor_toks)

for cid, c in enumerate(clusters, start=1):
    members = [items_sorted[idx] for idx in c["members"]]
    total_count = sum(m["count"] for m in members)
    # pick canonical display as most frequent / shortest
    candidates_sorted = sorted(members, key=lambda m: (-m["count"], len(m["label"])))
    canonical_display = candidates_sorted[0]["label"]
    # determine union tokens & majority value_type
    union_toks = set().union(*[m["toks"] for m in members])
    vt_counts = Counter(m["vt"] for m in members)
    maj_vt = vt_counts.most_common(1)[0][0] if vt_counts else "TEXT"

    # Try to attach to existing anchors by highest Jaccard over tokens
    best_anchor = None
    best_score = 0.0
    for a in ANCH:
        # Optionally restrict by value_type; here we use a soft filter
        score = score_tok_to_anchor(union_toks, a["tokset"])
        if score > best_score:
            best_score = score
            best_anchor = a

    # Decision threshold for using an existing anchor
    use_existing = best_anchor is not None and best_score >= 0.55

    if use_existing:
        # map every member to existing anchor
        for m in members:
            mapped_rows.append({
                "label": m["label"],
                "count": m["count"],
                "cluster_id": cid,
                "anchor_key": best_anchor["key"],
                "anchor_display": best_anchor["display"],
                "anchor_category": best_anchor["category"],
                "value_type": m["vt"],
                "reason": f"cluster->existing_anchor jaccard={best_score:.2f}",
                "normalized": m["norm"]
            })
    else:
        # propose a new anchor
        category = category_guess(union_toks)
        key = to_upper_snake(canonical_display)
        # ensure key is not colliding with existing
        if any(a["key"] == key for a in ANCH):
            key = f"{key}_NEW"
        new_anchor_rows.append({
            "cluster_id": cid,
            "canonical_key": key,
            "canonical_display": canonical_display,
            "category": category,
            "value_type": maj_vt,
            "total_count": total_count,
            "tokens": " ".join(sorted(union_toks)),
            "examples": " | ".join([m["label"] for m in candidates_sorted[:6]])
        })
        for m in members:
            mapped_rows.append({
                "label": m["label"],
                "count": m["count"],
                "cluster_id": cid,
                "anchor_key": key,
                "anchor_display": canonical_display,
                "anchor_category": category,
                "value_type": m["vt"],
                "reason": "cluster->new_anchor",
                "normalized": m["norm"]
            })

# Build DataFrames
mapped_df = pd.DataFrame(mapped_rows).sort_values(
    ["anchor_category","anchor_key","cluster_id","count"], ascending=[True,True,True,False]
)
new_anchors_df = pd.DataFrame(new_anchor_rows).sort_values(
    ["category","total_count"], ascending=[True,False]
)

# Some labels might have empty tokens and form singletons; keep them as still if needed
# But we already mapped all clusters to either existing or proposed anchors.
# However, guard: if a label is empty after strip(), send to still.
for m in items_sorted:
    if not m["label"]:
        still_rows.append({"label": m["label"], "count": m["count"], "note":"empty_label"})

still_df = pd.DataFrame(still_rows)

# Write outputs
out_mapped = OUTDIR / "auto_cluster_mapped_labels.csv"
out_newanchors = OUTDIR / "auto_cluster_new_anchors.csv"
out_still = OUTDIR / "auto_cluster_still_unassigned.csv"

mapped_df.to_csv(out_mapped, index=False, encoding="utf-8-sig", lineterminator="\n")
new_anchors_df.to_csv(out_newanchors, index=False, encoding="utf-8-sig", lineterminator="\n")
still_df.to_csv(out_still, index=False, encoding="utf-8-sig", lineterminator="\n")

print("Wrote:", out_mapped)
print("Wrote:", out_newanchors)
print("Wrote:", out_still)
