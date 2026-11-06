"""Publication checks reject stale, partially propagated and unavailable data."""

from contextlib import redirect_stdout
from copy import deepcopy
from http.client import IncompleteRead
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from threading import Thread
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import verify_publication as publication


class PublicationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.site = Path(directory.name)
        (self.site / "data").mkdir()
        self.contents = {name: b'{"value":1}' for name in publication.FILES}
        self.contents["data/health.json"] = json.dumps({
            "snapshotAt": "2026-11-06T15:00:00Z",
            "forecast": {"issuedAt": "2026-11-06T15:00:30Z"},
        }).encode()
        for name, content in self.contents.items():
            (self.site / name).write_bytes(content)
        self.manifest = publication.create_manifest(self.site)

    def serve(self, request, **kwargs):
        name = request.full_url.removeprefix("https://example.test/open-current/")
        self.assertNotIn("?", name)
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(request.get_header("Cache-control"), "no-cache")
        return io.BytesIO(self.contents[name])

    def test_uploaded_bytes_match_without_modifying_observation_times(self):
        self.assertEqual(self.manifest["snapshotAt"], "2026-11-06T15:00:00Z")
        self.assertEqual(self.manifest["forecastIssuedAt"], "2026-11-06T15:00:30Z")
        with patch.object(publication, "urlopen", side_effect=self.serve):
            self.assertEqual(publication.check_once(self.manifest, "https://example.test/open-current/"), [])

    def test_fresh_health_cannot_hide_a_stale_forecast_of_the_same_size(self):
        self.contents["data/forecasts.json"] = b'{"value":0}'
        with patch.object(publication, "urlopen", side_effect=self.serve):
            errors = publication.check_once(self.manifest, "https://example.test/open-current")
        self.assertEqual(len(errors), 1)
        self.assertIn("data/forecasts.json: public checksum differs", errors[0])

    def test_each_public_file_must_match(self):
        for name in publication.FILES:
            with self.subTest(name=name):
                original = self.contents[name]
                self.contents[name] = b"stale"
                with patch.object(publication, "urlopen", side_effect=self.serve):
                    errors = publication.check_once(self.manifest, "https://example.test/open-current/")
                self.contents[name] = original
                self.assertEqual(len(errors), 1)
                self.assertIn(name, errors[0])

    def test_http_errors_timeouts_and_incomplete_bodies_do_not_pass(self):
        for error in [HTTPError("https://example.test", 503, "unavailable", {}, None),
                      TimeoutError("timed out"), IncompleteRead(b"partial", 30)]:
            with self.subTest(error=error), patch.object(publication, "urlopen", side_effect=error):
                self.assertEqual(len(publication.check_once(self.manifest, "https://example.test/")), 4)

    def test_oversized_response_is_not_read_without_a_bound(self):
        response = io.BytesIO(b"x" * 1000)
        with patch.object(response, "read", wraps=response.read) as read, patch.object(
            publication, "urlopen", return_value=response,
        ):
            error = publication.check_file("https://example.test/", "data/index.json", self.manifest["files"]["data/index.json"])
        read.assert_called_once_with(len(self.contents["data/index.json"]) + 1)
        self.assertIn("size differs", error)

    def test_propagation_can_recover_within_the_polling_budget(self):
        with patch.object(publication, "check_once", side_effect=[["old snapshot"], []]) as check, patch.object(
            publication.time, "sleep",
        ) as sleep, redirect_stdout(io.StringIO()):
            self.assertTrue(publication.verify(self.manifest, "https://example.test", attempts=3, interval=30))
        self.assertEqual(check.call_count, 2)
        sleep.assert_called_once_with(30)

    def test_persistent_staleness_exhausts_the_budget_and_fails(self):
        with patch.object(publication, "check_once", return_value=["old snapshot"]) as check, patch.object(
            publication.time, "sleep",
        ) as sleep, redirect_stdout(io.StringIO()):
            self.assertFalse(publication.verify(self.manifest, "https://example.test", attempts=3, interval=30))
        self.assertEqual(check.call_count, 3)
        self.assertEqual(sleep.call_count, 2)

    def test_invalid_or_incomplete_manifests_cannot_report_success(self):
        bad_digest = deepcopy(self.manifest)
        bad_digest["files"]["data/index.json"]["sha256"] = "invalid"
        missing_file = deepcopy(self.manifest)
        del missing_file["files"]["data/forecasts.json"]
        for manifest in [{}, [], {"schema": 1, "files": {}}, bad_digest, missing_file]:
            with self.subTest(manifest=manifest):
                path = self.site / "manifest.json"
                path.write_text(json.dumps(manifest))
                with self.assertRaises(ValueError):
                    publication.load_manifest(path)

    def test_cli_checks_real_http_and_returns_nonzero_for_stale_data(self):
        contents = self.contents

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                content = contents[self.path.removeprefix("/open-current/")]
                self.send_response(200)
                self.send_header("Content-Length", str(len(content)))
                self.end_headers()
                self.wfile.write(content)

            def log_message(self, *_args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            script = str(ROOT / "scripts/verify_publication.py")
            manifest = self.site / "manifest.json"
            subprocess.run([sys.executable, script, "manifest", "--site-dir", str(self.site),
                            "--output", str(manifest)], check=True, capture_output=True)
            command = [sys.executable, script, "check", "--manifest", str(manifest),
                       "--url", f"http://127.0.0.1:{server.server_port}/open-current/", "--attempts", "1"]
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stderr)
            contents["data/forecasts.json"] = b'{"value":0}'
            result = subprocess.run(command, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn("data/forecasts.json: public checksum differs", result.stdout)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()
