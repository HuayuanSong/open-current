"""Download only the pinned model bundle and verify it before loading weights."""
import hashlib
import json
from pathlib import Path
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/'forecast/model-release.json').read_text())
DIRECTORY=ROOT/'.cache/forecast-serving/models'/CONFIG['version']

def verify(directory=DIRECTORY):
    manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['version']!=CONFIG['version']: raise ValueError('Unexpected model version')
    for name,digest in manifest['files'].items():
        path=directory/name
        if not path.is_relative_to(directory) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Model integrity mismatch: {name}')
    return directory

def ensure():
    if (DIRECTORY/'manifest.json').exists(): return verify()
    DIRECTORY.parent.mkdir(parents=True,exist_ok=True)
    archive=DIRECTORY.parent/'download.tar.gz'
    with urllib.request.urlopen(CONFIG['url'],timeout=120) as response:
        archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=CONFIG['sha256']:
        raise ValueError('Model release checksum mismatch')
    with tarfile.open(archive) as source:
        source.extractall(DIRECTORY.parent,filter='data')
    archive.unlink()
    return verify()

if __name__=='__main__': print(ensure())
