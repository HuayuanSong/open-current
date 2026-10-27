"""Pinned challenger bundle, independent of the public production model."""
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def ensure():
    config=json.loads((ROOT/'forecast/trial/model-release.json').read_text())
    directory=ROOT/'.cache/forecast-serving/trial-models'/config['version']
    if not (directory/'manifest.json').exists():
        directory.parent.mkdir(parents=True,exist_ok=True)
        archive=directory.parent/'download.tar.gz'
        with urllib.request.urlopen(config['url'],timeout=120) as response:archive.write_bytes(response.read())
        if digest(archive)!=config['sha256']:raise ValueError('Trial archive checksum mismatch')
        with tarfile.open(archive) as source:source.extractall(directory.parent,filter='data')
        archive.unlink()
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['version']!=config['version']:raise ValueError('Trial version mismatch')
    if manifest['protocolSha256']!=digest(ROOT/'forecast/trial/protocol.json'):raise ValueError('Frozen trial protocol changed')
    for name,expected in manifest['files'].items():
        path=directory/name
        if not path.is_relative_to(directory) or digest(path)!=expected:raise ValueError(f'Trial integrity mismatch: {name}')
    return directory
