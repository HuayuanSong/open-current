"""The credential form rejects cross-origin requests and does not log secrets."""

from pathlib import Path
import socket
import stat
import subprocess
import sys
import tempfile
import unittest
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


class SetupFormTests(unittest.TestCase):
    def test_origin_check_permissions_and_secret_redaction(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "secrets.json"
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            process = subprocess.Popen(
                [sys.executable, str(ROOT / "watchdog/setup_token.py"), "--output", str(destination), "--port", str(port)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            )
            try:
                line = process.stdout.readline().strip()
                self.assertTrue(line.startswith("Open "), line)
                url = line[5:]
                with urllib.request.urlopen(url, timeout=5) as response:
                    self.assertEqual(response.headers["Referrer-Policy"], "same-origin")
                token = "github_pat_" + "a" * 70
                body = urllib.parse.urlencode({"token": token}).encode()
                bad = urllib.request.Request(url, data=body, headers={"Origin": "https://unrelated.example"})
                with self.assertRaises(urllib.error.HTTPError) as error:
                    urllib.request.urlopen(bad, timeout=5)
                self.assertEqual(error.exception.code, 403)
                self.assertFalse(destination.exists())
                good = urllib.request.Request(url, data=body, headers={"Origin": f"http://127.0.0.1:{port}"})
                with urllib.request.urlopen(good, timeout=5) as response:
                    self.assertIn("Saved securely", response.read().decode())
                stdout, stderr = process.communicate(timeout=5)
                self.assertEqual(process.returncode, 0, stderr)
                self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
                self.assertNotIn(token, stdout + stderr)
            finally:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
