"""Quick script to fetch a URL and extract text with resiliparse."""

import sys
import time

import requests
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.html import HTMLTree

URL = "https://cs.stackexchange.com/questions/100026/ways-to-make-change-for-a-dollar-how-to-optimize-with-constraints"
OUTPUT = "cs_stackexchange_extract.txt"
MAX_RETRIES = 10
BACKOFF = 2  # seconds, doubles each retry


def fetch_with_retries(url, max_retries=MAX_RETRIES):
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt + 1}/{max_retries}...")
            resp = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
            resp.raise_for_status()
            print(f"Success! ({resp.status_code})")
            return resp.text
        except (requests.RequestException, requests.HTTPError) as e:
            wait = BACKOFF * (2 ** attempt)
            print(f"  Failed: {e}")
            if attempt < max_retries - 1:
                print(f"  Retrying in {wait}s...")
                time.sleep(wait)
    print("All retries exhausted.")
    sys.exit(1)


def main():
    html = fetch_with_retries(URL)
    tree = HTMLTree.parse(html)
    text = extract_plain_text(tree, main_content=True, preserve_formatting=True)
    with open(OUTPUT, "w") as f:
        f.write(text)
    print(f"\nExtracted {len(text)} chars -> {OUTPUT}")


if __name__ == "__main__":
    main()
