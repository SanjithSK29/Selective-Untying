"""Run provenance and atomic artifact writes (standard library only)."""
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


def provenance(args, inputs):
    root = Path(__file__).resolve().parent
    config = {k: v for k, v in vars(args).items()
              if k not in {'out', 'artifact_dir', 'methods', 'seeds', 'alphas', 'resume'}}
    versions = {}
    for package in ('torch', 'numpy', 'tokenizers', 'tqdm', 'huggingface-hub', 'datasets'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root,
                                         text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    import torch
    return {'schema_version': 1, 'config': config,
            'input_sha256': {str(p): sha256(p) for p in inputs},
            'source_sha256': {p.name: sha256(p) for p in sorted(root.glob('*.py'))},
            'git_commit': commit, 'python': platform.python_version(),
            'packages': versions, 'device': torch.cuda.get_device_name(0)
            if torch.cuda.is_available() else 'cpu',
            'cuda': torch.version.cuda}


def prepare_results(path, manifest):
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + '.manifest.json')
    if path.exists():
        if not sidecar.exists():
            raise ValueError('Existing results lack provenance. Use a new --out path; legacy results are preserved.')
        if json.loads(sidecar.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Run configuration, inputs, source, or environment changed. Use a new --out path.')
        return json.loads(path.read_text(encoding='utf-8'))
    atomic_json(sidecar, manifest)
    return {}


def save_model(path, model, manifest, **metadata):
    import torch
    from dataclasses import asdict
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    torch.save({'model': model.state_dict(), 'config': asdict(model.config),
                'output_mode': model.output_mode, 'manifest': manifest,
                **metadata}, temporary)
    os.replace(temporary, path)


def prepare_artifacts(directory, manifest):
    directory = Path(directory)
    marker = directory / "manifest.json"
    if directory.exists() and any(directory.iterdir()):
        if not marker.exists() or json.loads(marker.read_text(encoding="utf-8")) != manifest:
            raise ValueError("Artifact directory belongs to a different or unverified run. Use a new --artifact_dir.")
    atomic_json(marker, manifest)
