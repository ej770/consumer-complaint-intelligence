"""Step 7 - Build the self-contained interactive dashboard (dashboard/index.html).

Headline findings are written from the result tables here, so every number on the page comes
straight from the pipeline outputs.
"""
import json
import sys
from pathlib import Path

import pandas as pd

import config as C


def T(name):
    return pd.read_csv(C.TABLES / f"{name}.csv")


def findings() -> list:
    comp = T("03_company_benchmark").set_index("company")
    exp_ = comp.loc["Experian Information Solutions Inc.", "relief_rate_pct"]
    tu = comp.loc["TRANSUNION INTERMEDIATE HOLDINGS, INC.", "relief_rate_pct"]

    themes = T("theme_summary").set_index("theme")
    it = themes.loc["Identity theft"]
    it_gap = 100 * (it["relief_rate"] - it["expected_relief_given_products"])
    mort = themes.loc["Mortgage servicing & escrow", "relief_rate"]

    sig = T("05_language_signals").set_index("signal")
    court = sig.loc["Mentions suing or court", "mix_adjusted_gap_pts"]
    tone = T("sentiment_vs_relief")
    q1, q5 = tone.iloc[0]["relief_rate"], tone.iloc[4]["relief_rate"]

    auto = T("routing_automation").dropna()
    ok = auto[auto["accuracy_on_auto_routed"] >= 0.95].iloc[0]
    relief = T("relief_model_comparison")
    cap = relief[relief["model"].str.startswith("Combined")]["capture_top20_pct"].iloc[0]

    return [
        {"title": "Who you complain to matters",
         "text": f"Experian closed {exp_:.1f}% of its complaints with relief. TransUnion, answering the "
                 f"same kind of credit-report disputes, closed {tu:.1f}%."},
        {"title": "What went wrong matters",
         "text": f"Identity-theft complaints ended in relief {it['relief_rate']:.1%} of the time, "
                 f"{it_gap:.1f} points above what their product mix predicts. Mortgage servicing: {mort:.1%}."},
        {"title": "How it is written barely does",
         "text": f"Complaints that mention suing or court got relief {abs(court):.1f} points less often than "
                 f"expected. The most negative fifth of narratives fared like the least negative "
                 f"({q1:.1%} vs. {q5:.1%})."},
        {"title": "A model can take the first pass",
         "text": f"The routing model sends {ok['coverage']:.0%} of complaints to the right team automatically "
                 f"at {ok['accuracy_on_auto_routed']:.0%} accuracy. The relief model's top fifth of the queue "
                 f"holds {cap:.0%} of relief cases."},
    ]


def main() -> None:
    data = json.loads((C.DASHBOARD / "dashboard_data.json").read_text())
    data["findings"] = findings()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    template = (C.DASHBOARD / "template.html").read_text()
    assert "/*__DATA__*/null" in template
    body = template.replace("/*__DATA__*/null", payload)
    # Standalone page (open locally, upload to any website, or host on GitHub Pages):
    # title, meta, font links and styles go in <head>; the page itself goes in <body>.
    split = body.index("</style>") + len("</style>")
    head, page = body[:split], body[split:]
    html = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            '<meta property="og:title" content="Consumer Complaint Intelligence">\n'
            '<meta property="og:description" content="AI triage and relief analytics on 117,214 CFPB consumer complaints.">\n'
            '<style>body{margin:0}[hidden]{display:none!important}</style>\n'
            + head + '\n</head>\n<body>\n' + page + '\n</body>\n</html>\n')
    out = C.DASHBOARD / "index.html"
    out.write_text(html)
    if len(sys.argv) > 1:                       # optional: write the bare page body as well
        Path(sys.argv[1]).write_text(body)
    print(f"Dashboard -> {out.relative_to(C.ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")
    for f in data["findings"]:
        print(f" - {f['title']}: {f['text']}")


if __name__ == "__main__":
    main()
