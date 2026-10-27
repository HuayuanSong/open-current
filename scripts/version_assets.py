"""Fingerprint first-party assets, including the JS module dependency."""
import hashlib
from pathlib import Path
import re

DIST = Path(__file__).resolve().parents[1] / 'dist'


def digest(name):
    return hashlib.sha256((DIST / name).read_bytes()).hexdigest()[:12]


def main():
    for script, dependency in [('app.js', 'model.js'), ('forecasts.js', 'forecast-model.js')]:
        app = DIST / script
        text = re.sub(r'\./' + re.escape(dependency) + r'(?:\?v=[a-f0-9]+)?',
                      './' + dependency + '?v=' + digest(dependency), app.read_text())
        app.write_text(text)
    page = DIST / 'index.html'
    text = page.read_text()
    for name in ['style.css', 'app.js', 'forecasts.js', 'icon.svg']:
        text = re.sub(re.escape(name) + r'(?:\?v=[a-f0-9]+)?', name + '?v=' + digest(name), text)
    page.write_text(text)
    print('Versioned app, stylesheet, icon and module imports.')


if __name__ == '__main__':
    main()

