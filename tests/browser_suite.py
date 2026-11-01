"""Run all browser checks against one temporary local server or GRID_TEST_URL."""

from functools import partial
from http.server import ThreadingHTTPServer
import os
from pathlib import Path
import subprocess
import sys
import threading

from browser_regressions import QuietHandler

ROOT = Path(__file__).resolve().parents[1]


def run_checks(url):
    environment = {**os.environ, "GRID_TEST_URL": url.rstrip("/") + "/"}
    for name in (
        "browser_check.py",
        "forecast_browser_check.py",
        "browser_regressions.py",
    ):
        subprocess.run(
            [sys.executable, str(ROOT / "tests" / name)],
            cwd=ROOT,
            env=environment,
            check=True,
        )


def main():
    if url := os.environ.get("GRID_TEST_URL"):
        run_checks(url)
        return
    with ThreadingHTTPServer(
        ("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "dist"))
    ) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            run_checks(f"http://127.0.0.1:{server.server_port}/")
        finally:
            server.shutdown()
            thread.join(timeout=5)


if __name__ == "__main__":
    main()
