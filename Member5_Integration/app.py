"""
app.py - Phishing URL checker (command-line interface).

Usage:
    python app.py --url "http://paypal.secure-login.xyz/verify"
    python app.py --file sample_urls.txt          # batch scan
    python app.py --interactive                   # type URLs one by one
    python app.py --demo                          # scripted demo for the recording

Flow:  user input -> validation (Member 2) -> saved model (Member 3)
       -> alert + preventive action (the simulated "Open link" is disabled
          for flagged URLs) -> logged to logs/alerts_log.csv
"""
import argparse
import csv
import os
import sys
from datetime import datetime

from predictor import PhishingDetector

LOG_PATH = os.path.join(os.path.dirname(__file__), "logs", "alerts_log.csv")
USE_COLOR = sys.stdout.isatty()
RED, GREEN, YELLOW, BOLD, END = (("\033[91m", "\033[92m", "\033[93m", "\033[1m", "\033[0m")
                                 if USE_COLOR else ("", "", "", "", ""))


class SimulatedBrowser:
    """Stand-in for a real browser/email client. Nothing is ever opened."""

    def open_link(self, result):
        if result["label"] == "Phishing":
            return False, "BLOCKED - 'Open link' is disabled for this URL."
        if result["label"] == "Invalid":
            return False, "BLOCKED - input could not be analysed."
        return True, "Link opened (simulated)."


def log_result(result, opened):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["timestamp", "url", "label", "probability", "link_opened"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), result["url"],
                    result["label"], f"{result['probability']:.3f}", opened])


def show(result, browser):
    """Print the verdict, show the alert if needed, apply the preventive action."""
    opened, action_msg = browser.open_link(result)
    pct = result["probability"] * 100
    print("-" * 64)
    print(f"URL: {result['url']}")
    if result["label"] == "Phishing":
        print(f"{RED}{BOLD}!! WARNING: PHISHING SUSPECTED ({pct:.1f}% confidence) !!{END}")
        for r in result["reasons"] or ["No single rule matched, but the model scored this host name as high risk"]:
            print(f"   - {r}")
        print(f"[ Open link ]  {RED}{action_msg}{END}")
        print(f"   {YELLOW}Do not enter passwords or personal details. Report to IT/security.{END}")
    elif result["label"] == "Legitimate":
        print(f"{GREEN}OK: Looks legitimate (phishing probability {pct:.1f}%){END}")
        print(f"[ Open link ]  {action_msg}")
    else:
        print(f"{YELLOW}Could not analyse: {'; '.join(result['reasons'])}{END}")
        print(f"[ Open link ]  {action_msg}")
    log_result(result, opened)


def scan_file(detector, browser, path):
    with open(path, encoding="utf-8") as f:
        urls = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    counts = {"Phishing": 0, "Legitimate": 0, "Invalid": 0}
    for u in urls:
        r = detector.predict(u)
        counts[r["label"]] += 1
        show(r, browser)
    print("=" * 64)
    print(f"Scanned {len(urls)} URLs: {counts['Phishing']} phishing, "
          f"{counts['Legitimate']} legitimate, {counts['Invalid']} invalid.")


def interactive(detector, browser):
    print("Phishing URL checker - type a URL (or 'quit').")
    while True:
        try:
            u = input("\nURL> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in {"q", "quit", "exit"}:
            break
        show(detector.predict(u), browser)


def demo(detector, browser):
    print(f"{BOLD}=== DEMO PART 1: simulated email with 4 links ==={END}")
    print('"Dear customer, your account has been suspended. Click below to restore access."\n')
    for u in ["http://appleid.apple.com.sign-inx9p3fygrfpyvzdkn.top/",
              "http://coinbase.com.ticket-94.id/",
              "https://www.watson.ch",
              "https://www.freshplanet.com"]:
        show(detector.predict(u), browser)
    print(f"\n{BOLD}=== DEMO PART 2: invalid input ==={END}")
    for u in ["not a url ::", "www.google.com"]:
        show(detector.predict(u), browser)
    print(f"\n{BOLD}=== DEMO PART 3: known limitations (shown on purpose) ==={END}")
    print("A phishing URL the model misses (false negative), and a real Google page it wrongly flags (false positive):\n")
    for u in ["https://sfr-commande-annulation.com/verification/loading.php",
              "https://accounts.google.com/signin"]:
        show(detector.predict(u), browser)
    print("=" * 64)
    print("Demo finished. All decisions were written to logs/alerts_log.csv")


def main():
    ap = argparse.ArgumentParser(description="AI-powered phishing URL checker")
    ap.add_argument("--url", help="check a single URL")
    ap.add_argument("--file", help="text file with one URL per line")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5, help="phishing probability cut-off")
    a = ap.parse_args()

    detector = PhishingDetector(threshold=a.threshold)
    browser = SimulatedBrowser()

    if a.url:
        show(detector.predict(a.url), browser)
    elif a.file:
        scan_file(detector, browser, a.file)
    elif a.interactive:
        interactive(detector, browser)
    elif a.demo:
        demo(detector, browser)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
