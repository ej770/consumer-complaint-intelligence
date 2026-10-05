"""Shared paths and settings for the CFPB Complaint Intelligence project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "complaints.csv.gz"
PROCESSED = ROOT / "data" / "processed"
CLEAN = PROCESSED / "complaints_clean.csv.gz"
DB = PROCESSED / "complaints.db"
SQL_DIR = ROOT / "sql"
TABLES = ROOT / "outputs" / "tables"
FIGURES = ROOT / "outputs" / "figures"
POWERBI = ROOT / "powerbi"
DASHBOARD = ROOT / "dashboard"

# Source: CFPB Consumer Complaint Database extract (complaints received 2019-01-01 to 2020-01-09,
# all with published consumer narratives), as distributed with "Supervised Machine Learning for
# Text Analysis in R" (Hvitfeldt & Silge): https://github.com/EmilHvitfeldt/smltar
# Pinned to a commit and checked against a checksum, so every run analyses the same file.
RAW_URL = ("https://github.com/EmilHvitfeldt/smltar/raw/"
           "3ae4ab3e7ce53e3d74dded5f17e7cf8dbe164caa/data/complaints.csv.gz")
RAW_SHA256 = "f4a5e174d364f1e1a25fac324e1a8224ff4e9454326e6e58abaa194bca93097b"

RANDOM_STATE = 42

PRODUCT_SHORT = {
    "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting",
    "Debt collection": "Debt collection",
    "Credit card or prepaid card": "Credit card / prepaid",
    "Mortgage": "Mortgage",
    "Checking or savings account": "Checking / savings",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan / lease",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Payday loan, title loan, or personal loan": "Payday / personal loan",
}

# Short display names for the 15 highest-volume companies (dashboard labels only)
COMPANY_DISPLAY = {
    "TRANSUNION INTERMEDIATE HOLDINGS, INC.": "TransUnion",
    "EQUIFAX, INC.": "Equifax",
    "Experian Information Solutions Inc.": "Experian",
    "CITIBANK, N.A.": "Citibank",
    "CAPITAL ONE FINANCIAL CORPORATION": "Capital One",
    "JPMORGAN CHASE & CO.": "JPMorgan Chase",
    "BANK OF AMERICA, NATIONAL ASSOCIATION": "Bank of America",
    "WELLS FARGO & COMPANY": "Wells Fargo",
    "SYNCHRONY FINANCIAL": "Synchrony Financial",
    "Navient Solutions, LLC.": "Navient",
    "AMERICAN EXPRESS COMPANY": "American Express",
    "Alliance Data Card Services": "Alliance Data Card Services",
    "AES/PHEAA": "AES/PHEAA",
    "PORTFOLIO RECOVERY ASSOCIATES INC": "Portfolio Recovery Associates",
    "U.S. BANCORP": "U.S. Bancorp",
    "All other companies": "All other companies",
}

# Greater Houston = USPS 3-digit ZIP prefixes 770-775 (CFPB masks the last two ZIP digits).
HOUSTON_ZIP3 = {"770", "771", "772", "773", "774", "775"}

# Months with complete narrative publication. Narratives are published with a lag, so
# Nov 2019 - Jan 2020 are only partially represented in this extract.
COMPLETE_MONTHS = ("2019-01", "2019-10")

# Generic complaint words plus company names: we want themes about *what went wrong*,
# not *which company* (company is analysed separately).
STOP_EXTRA = {
    "dollaramount", "did", "does", "don", "just", "like", "told", "called", "said", "got", "asked",
    "im", "ive", "didnt", "dont", "wa", "would", "also", "could", "still", "get", "one", "will",
    "time", "received", "sent", "back", "even", "never", "day", "days",
    "wells", "fargo", "wf", "chase", "jpmorgan", "america", "boa", "capital", "citi", "citibank",
    "synchrony", "navient", "experian", "equifax", "transunion", "trans", "union", "amex",
    "american", "express", "discover", "paypal", "ocwen", "nationstar", "cooper", "santander",
    "ally", "pheaa", "fedloan", "sallie", "mae", "barclays", "comenity", "sun", "trust", "suntrust",
    "usaa", "td", "pnc", "regions", "huntington", "portfolio", "recovery", "midland", "encore",
    "convergent", "erc", "ic", "lvnv", "cavalry", "transworld", "coinbase", "venmo", "cash", "app",
    "square", "navy", "federal", "loancare", "caliber", "quicken", "rocket", "shellpoint",
    "carrington", "seterus", "ditech", "bayview", "selene", "penfed", "alliance",
}
