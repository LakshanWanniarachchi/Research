"""Capture the running web application, page by page.

Screenshots for the thesis have to come from the application as it actually
runs, against the real database, signed in as a real account. So this drives
the installed Chrome through Playwright rather than mocking anything: it signs
in through the ordinary login form and photographs whatever the server sends
back. If a page is broken, the screenshot shows it broken.

    python tools/screenshot_ui.py --password '...' [--out ../figures/ui]

Chrome is driven through `channel="chrome"`, so no separate browser download
is needed.
"""
import argparse
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = "http://127.0.0.1:5000"

# (filename, path, wait) - `wait` is extra settling time in seconds for pages
# that draw a canvas or poll before they have anything to show.
PAGES = [
    ("01_login",      "/login",        0.4),
    ("02_dashboard",  "/dashboard",    0.8),
    ("03_live_wifi",  "/ecg/live",     2.5),
    ("04_live_usb",   "/live",         2.0),
    ("05_history",    "/ecg/history",  0.6),
    ("06_devices",    "/devices",      0.6),
    ("07_profile",    "/profile",      0.6),
    ("08_replay",     "/",             1.4),
]

VIEWPORTS = [("desktop", 1440, 900), ("mobile", 390, 844)]


def newest_report_path(page):
    """The report page needs a real report id, so take the newest in history."""
    page.goto(BASE + "/ecg/history", wait_until="load")
    link = page.query_selector("tbody a[href^='/ecg/']")
    return link.get_attribute("href") if link else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="admin")
    ap.add_argument("--password", required=True)
    ap.add_argument("--out", default=os.path.join("..", "figures", "ui"))
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    problems = []

    with sync_playwright() as p:
        browser = p.chromium.launch(channel="chrome")
        for label, width, height in VIEWPORTS:
            ctx = browser.new_context(viewport={"width": width, "height": height},
                                      device_scale_factor=2)
            page = ctx.new_page()
            # Console errors are collected rather than ignored: a page that
            # looks right in a screenshot and throws on every poll is not a
            # page that works.
            page.on("console", lambda m: problems.append(
                (label, page.url, m.type, m.text)) if m.type == "error" else None)
            page.on("pageerror", lambda e: problems.append((label, page.url, "exception", str(e))))

            # Sign in once per viewport.
            page.goto(BASE + "/login", wait_until="load")
            page.screenshot(path=os.path.join(out, f"01_login_{label}.png"), full_page=True)
            page.fill("#username", args.user)
            page.fill("#password", args.password)
            page.click("button[type=submit]")
            page.wait_for_load_state("networkidle")
            if "/login" in page.url:
                print("  sign-in failed - check the password", file=sys.stderr)
                return 1

            for name, path, wait in PAGES:
                if path == "/login":
                    continue
                page.goto(BASE + path, wait_until="load")
                time.sleep(wait)
                page.screenshot(path=os.path.join(out, f"{name}_{label}.png"), full_page=True)
                print(f"  {name}_{label}.png")

            report = newest_report_path(page)
            if report:
                page.goto(BASE + report, wait_until="load")
                time.sleep(0.8)
                page.screenshot(path=os.path.join(out, f"09_report_{label}.png"), full_page=True)
                print(f"  09_report_{label}.png  ({report})")
            else:
                print("  no stored report to photograph", file=sys.stderr)

            ctx.close()
        browser.close()

    if problems:
        print("\nConsole problems:")
        for row in problems:
            print("  ", row)
    else:
        print("\nNo console errors on any page.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
