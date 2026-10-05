"""Step 1 - Load the raw CFPB extract, clean it and engineer intake-time features.

Output: data/processed/complaints_clean.csv.gz (one row per complaint)
"""
import hashlib
import re
import urllib.request

import numpy as np
import pandas as pd

import config as C

RELIEF = {"Closed with monetary relief", "Closed with non-monetary relief"}

# Language signals found in the consumer's own narrative (all available at intake).
SIGNALS = {
    "mentions_attorney": r"\b(?:attorney|lawyer|legal counsel|law firm)s?\b",
    "mentions_legal_action": r"\b(?:sue|suing|lawsuit|litigation|small claims|court)\b",
    "mentions_regulator": r"\b(?:cfpb|attorney general|ftc|federal trade commission|better business bureau|bbb)\b",
    "cites_law": r"(?:\bfcra\b|\bfdcpa\b|fair credit reporting act|fair debt collection|\b15 ?u\.?s\.?c|\b1681\w*|\b1692\w*)",
    "mentions_identity_theft": r"(?:identity theft|identity was stolen|stolen identity|not my account|fraudulent account)",
    "mentions_fees": r"\b(?:fee|fees|overdraft|interest charge)\b",
}


def clean_text(s: pd.Series) -> pd.Series:
    """Lower-case narratives and strip CFPB redaction masks (XXXX, {$0.00}) and extra whitespace."""
    s = s.fillna("").str.lower()
    s = s.str.replace(r"x{2,}", " ", regex=True)          # redacted names/dates/accounts
    s = s.str.replace(r"\{\$[\d,\.]+\}", " dollaramount ", regex=True)  # redacted dollar amounts
    s = s.str.replace(r"[^a-z\s']", " ", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def main() -> None:
    if not C.RAW.exists():
        C.RAW.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading raw data from {C.RAW_URL} ...")
        urllib.request.urlretrieve(C.RAW_URL, C.RAW)
    digest = hashlib.sha256(C.RAW.read_bytes()).hexdigest()
    if digest != C.RAW_SHA256:
        raise SystemExit(f"{C.RAW} is not the expected extract (sha256 {digest[:12]}...). "
                         "Delete it and run again to download a fresh copy.")

    raw = pd.read_csv(C.RAW)
    print(f"Raw rows: {len(raw):,}")

    df = pd.DataFrame({
        "complaint_id": raw["complaint_id"],
        "date_received": pd.to_datetime(raw["date_received"]),
        "product_full": raw["product"],
        "product": raw["product"].map(C.PRODUCT_SHORT),
        "sub_product": raw["sub_product"].fillna("(none)"),
        "issue": raw["issue"],
        "sub_issue": raw["sub_issue"].fillna("(none)"),
        "company": raw["company"].str.strip(),
        "state": raw["state"].fillna("Unknown"),
        "zip3": raw["zip_code"].fillna("").str[:3],
        "tags": raw["tags"].fillna(""),
        "company_response": raw["company_response_to_consumer"],
        "timely": (raw["timely_response"] == "Yes").astype(int),
        "date_sent_to_company": pd.to_datetime(raw["date_sent_to_company"]),
        "narrative": raw["consumer_complaint_narrative"].fillna(""),
    })
    assert df["product"].notna().all(), "Unmapped product label"

    # Outcome flags
    df["relief"] = df["company_response"].isin(RELIEF).astype(int)
    df["monetary_relief"] = (df["company_response"] == "Closed with monetary relief").astype(int)

    # Dates
    df["month"] = df["date_received"].dt.strftime("%Y-%m")
    df["days_to_company"] = (df["date_sent_to_company"] - df["date_received"]).dt.days
    df["complete_month"] = df["month"].between(*C.COMPLETE_MONTHS).astype(int)

    # Consumer segments and geography
    df["older_american"] = df["tags"].str.contains("Older American").astype(int)
    df["servicemember"] = df["tags"].str.contains("Servicemember").astype(int)
    df["is_texas"] = (df["state"] == "TX").astype(int)
    df["is_houston"] = df["zip3"].isin(C.HOUSTON_ZIP3).astype(int) * df["is_texas"]

    # Text features
    df["narrative_clean"] = clean_text(df["narrative"])
    df["narrative_words"] = df["narrative_clean"].str.split().str.len().fillna(0).astype(int)
    lower = df["narrative"].str.lower()
    for col, pattern in SIGNALS.items():
        df[col] = lower.str.contains(pattern, regex=True).astype(int)

    # Verbatim-duplicate narratives (same text filed more than once, e.g. against all three
    # credit bureaus, or form letters). Grouping by text hash keeps duplicates out of both
    # sides of any train/test split.
    df["text_group"] = pd.util.hash_pandas_object(df["narrative_clean"], index=False).astype("int64")
    df["dup_count"] = df.groupby("text_group")["complaint_id"].transform("size")
    df["is_duplicate_text"] = (df["dup_count"] > 1).astype(int)

    # Company grouping for reporting (top 15 by volume, rest = Other)
    top = df["company"].value_counts().head(15).index
    df["company_group"] = np.where(df["company"].isin(top), df["company"], "All other companies")

    df = df.drop(columns=["date_sent_to_company"])
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(C.CLEAN, index=False, compression="gzip")

    print(f"Clean rows: {len(df):,} | companies: {df['company'].nunique():,} | "
          f"relief rate: {df['relief'].mean():.1%} | duplicate-text share: {df['is_duplicate_text'].mean():.1%}")
    print(f"Saved -> {C.CLEAN.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
