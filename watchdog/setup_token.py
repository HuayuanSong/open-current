"""Collect a fine-grained GitHub token through a local, one-use browser form.

The output is a mode-0600 secrets file for `cf deploy --secrets-file`.
Delete it after deployment. Tokens never appear in command arguments or logs.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import re
import secrets
from urllib.parse import parse_qs, urlencode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("The secrets file already exists; use or remove it before starting setup again.")
    origin = f"http://127.0.0.1:{args.port}"
    route = "/" + secrets.token_urlsafe(32)
    link = "https://github.com/settings/personal-access-tokens/new?" + urlencode({
        "name": "Open Current watchdog",
        "description": "Trigger Open Current refresh and freshness-alert workflows",
        "target_name": "HuayuanSong", "expires_in": "90", "actions": "write",
    })
    saved = False

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *arguments):
            pass

        def reply(self, status, body):
            encoded = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Referrer-Policy", "same-origin")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def valid(self):
            return self.path == route and self.headers.get("Host") == f"127.0.0.1:{args.port}"

        def do_GET(self):
            if not self.valid():
                return self.reply(404, "Not found")
            self.reply(200, f'''<!doctype html><html lang="en"><meta charset="utf-8">
                <meta name="viewport" content="width=device-width,initial-scale=1">
                <title>Open Current · Scheduler setup</title>
                <style>body{{font:18px/1.6 system-ui;max-width:680px;margin:60px auto;padding:24px;color:#183744}}input,button{{font:inherit;padding:12px;box-sizing:border-box}}input{{width:100%}}button{{margin-top:18px;background:#185b51;color:white;border:0;border-radius:5px}}li{{margin-bottom:12px}}</style>
                <h1>Connect the backup scheduler</h1>
                <ol><li><a href="{link}" target="_blank" rel="noopener noreferrer">Create the GitHub token</a>.</li>
                <li>Under <b>Repository access</b>, select <b>Only select repositories</b>, then <b>open-current</b>.</li>
                <li>Check that the permission is <b>Actions: Read and write</b>. Metadata read access is automatic. No other permissions are needed.</li>
                <li>Generate the token and paste it below. The proposed expiry is 90 days; it will need renewal then.</li></ol>
                <p>This form runs only on your laptop. The token is saved temporarily with owner-only file permissions for deployment to Cloudflare.</p>
                <form method="post" action="{route}"><label for="token">Fine-grained GitHub token</label>
                <input id="token" name="token" type="password" autocomplete="off" required>
                <button type="submit">Save for deployment</button></form></html>''')

        def do_POST(self):
            nonlocal saved
            if not self.valid() or self.headers.get("Origin") != origin or saved:
                return self.reply(403, "Request rejected")
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length < 2048:
                    raise ValueError()
                token = parse_qs(self.rfile.read(length).decode()).get("token", [""])[0].strip()
                if not re.fullmatch(r"github_pat_[A-Za-z0-9_]{40,200}", token):
                    return self.reply(400, "Enter a fine-grained GitHub token. Use Back to try again.")
                args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
                with os.fdopen(descriptor, "w") as output:
                    json.dump({"GITHUB_TOKEN": token}, output)
                saved = True
                self.reply(200, "<h1>Saved securely</h1><p>You can close this tab. Tell Codex that the token is saved so deployment can finish.</p>")
                print("Token saved securely; no token value was logged.", flush=True)
            except (ValueError, OSError):
                self.reply(400, "Could not save the token. No token value was logged.")

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    server.timeout = 1
    print("Open " + origin + route, flush=True)
    while not saved:
        server.handle_request()
    server.server_close()


if __name__ == "__main__":
    main()
