"""Optional step 8 - Save a screenshot of the dashboard for the README.

Needs Playwright:  pip install playwright && python -m playwright install chromium
"""
import sys

import config as C

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sys.exit("Playwright is not installed; skipping the dashboard screenshot.")


def main() -> None:
    page_path = (C.DASHBOARD / "index.html").resolve()
    out = C.FIGURES / "dashboard_preview.png"
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 1040}, device_scale_factor=1.5)
        page.goto(page_path.as_uri())
        page.wait_for_timeout(1500)          # let fonts and charts settle
        page.screenshot(path=str(out), clip={"x": 0, "y": 0, "width": 1280, "height": 1040})
        browser.close()
    print(f"Screenshot -> {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()
