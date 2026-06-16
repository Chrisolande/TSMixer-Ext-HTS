#!/usr/bin/env python3
import os
import sys
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

port = os.environ.get("PORT", "8000")
url = f"http://localhost:{port}/healthz"

try:
    with urlopen(url, timeout=5) as resp:
        if resp.status != 200:
            sys.exit(1)
except (URLError, HTTPError, OSError):
    sys.exit(1)

sys.exit(0)