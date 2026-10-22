"""Verify delivery file hashes without loading model checkpoints."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
manifest = json.loads((root / 'delivery_manifest.json').read_text(encoding='utf-8'))
for record in manifest['files']:
    path = (root / record['path']).resolve()
    assert path.is_relative_to(root), record['path']
    assert path.is_file(), f'Missing: {record["path"]}'
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == record['sha256'], record['path']
print(f'OK: {len(manifest["files"])} files match delivery SHA-256 hashes.')
