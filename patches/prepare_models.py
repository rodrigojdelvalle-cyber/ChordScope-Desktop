from __future__ import annotations

import hashlib
import shutil
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "vendor_models"
MODELS.mkdir(parents=True, exist_ok=True)


def step(msg: str):
    print(f"\n=== {msg} ===", flush=True)


def prepare_btc():
    step("BTC Transformer")
    from huggingface_hub import snapshot_download
    target = MODELS / "btc-chord"
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id="puar-playground/btc-chord",
        local_dir=str(target),
        local_dir_use_symlinks=False,
        allow_patterns=[
            "config.json", "modeling_btc.py", "requirements.txt",
            "btc_model.pt", "btc_model_large_voca.pt", "btc_src/*",
        ],
    )
    print("BTC listo:", target)


def prepare_beat_this():
    step("Beat This! final0")
    import torch
    from beat_this.inference import load_checkpoint
    target = MODELS / "beat_this"
    target.mkdir(parents=True, exist_ok=True)
    dest = target / "beat_this-final0.ckpt"
    ckpt = load_checkpoint("final0", device="cpu")
    torch.save(ckpt, dest)
    print("Beat This listo:", dest, f"({dest.stat().st_size/1024/1024:.1f} MB)")


def _download_verified(url: str, dest: Path) -> None:
    """Download a Demucs checkpoint and verify its filename checksum prefix."""
    if dest.exists() and dest.stat().st_size > 1024:
        expected = dest.stem.split("-", 1)[1] if "-" in dest.stem else ""
        if expected:
            digest = hashlib.sha256(dest.read_bytes()).hexdigest()[: len(expected)]
            if digest == expected:
                print("Ya presente y verificado:", dest.name)
                return
        dest.unlink(missing_ok=True)

    print("Descargando:", dest.name)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=180) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, length=1024 * 1024)
    tmp.replace(dest)

    expected = dest.stem.split("-", 1)[1] if "-" in dest.stem else ""
    if expected:
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()[: len(expected)]
        if digest != expected:
            dest.unlink(missing_ok=True)
            raise RuntimeError(
                f"Checksum Demucs inválido para {dest.name}: esperado {expected}, obtenido {digest}"
            )
    print("Verificado:", dest.name, f"({dest.stat().st_size/1024/1024:.1f} MB)")


def prepare_demucs():
    step("Demucs HTDemucs-FT · repositorio local para Separator")
    from demucs_infer.api import Separator

    target = MODELS / "demucs"
    target.mkdir(parents=True, exist_ok=True)

    # htdemucs_ft is the official four-model fine-tuned ensemble. A local
    # Demucs repository consists of this bag YAML plus its four checkpoints.
    (target / "htdemucs_ft.yaml").write_text(
        "models: ['f7e0c4bc', 'd12395a8', '92cfc3b6', '04573f0d']\n"
        "weights:\n"
        "  - [1., 0., 0., 0.]\n"
        "  - [0., 1., 0., 0.]\n"
        "  - [0., 0., 1., 0.]\n"
        "  - [0., 0., 0., 1.]\n",
        encoding="utf-8",
    )

    root = "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/"
    files = [
        "f7e0c4bc-ba3fe64a.th",
        "d12395a8-e57c48e6.th",
        "92cfc3b6-ef3bcb9c.th",
        "04573f0d-f3cf25b2.th",
    ]
    for name in files:
        _download_verified(root + name, target / name)

    # Validate the exact stable API used by ChordScope. Loading the local bag
    # proves that all required weights resolve without Internet.
    sep = Separator(model="htdemucs_ft", repo=target, device="cpu", shifts=0, progress=False)
    if int(sep.samplerate) != 44100:
        raise RuntimeError(f"Samplerate Demucs inesperado: {sep.samplerate}")
    sources = list(getattr(sep.model, "sources", []))
    if sources and not {"drums", "bass", "other", "vocals"}.issubset(set(sources)):
        raise RuntimeError(f"Stems Demucs inesperados: {sources}")
    del sep
    print("Demucs local listo:", target)


def prepare_guitar_db():
    step("Guitar chord database")
    url = "https://raw.githubusercontent.com/tombatossals/chords-db/master/lib/guitar.json"
    dest = ROOT / "chordscope" / "data" / "guitar.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    print("Guitar DB listo:", dest, f"({dest.stat().st_size/1024:.1f} KB)")


def main():
    failures = []
    for name, fn in [
        ("BTC", prepare_btc),
        ("Beat This", prepare_beat_this),
        ("Demucs", prepare_demucs),
        ("Guitar DB", prepare_guitar_db),
    ]:
        try:
            fn()
        except Exception as exc:
            failures.append((name, exc))
            print(f"[ADVERTENCIA] {name}: {type(exc).__name__}: {exc}")
    print("\nResumen:")
    if failures:
        for name, exc in failures:
            print(" -", name, "NO preparado:", exc)
        print("\nLa aplicación aún puede ejecutarse con fallbacks, pero el EXE no será 100% offline.")
        return 2
    print("Todos los modelos y bases quedaron preparados para empaquetado offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
