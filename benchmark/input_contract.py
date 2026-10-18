from __future__ import annotations

import hashlib
import json
from pathlib import Path

SHARED_OBSERVATIONS = {
    "cell_metadata": "intermediate/metadata/cell_metadata.tsv",
    "rna_counts": "intermediate/rna/rna_counts_gene_by_cell.tsv",
    "atac_counts": "intermediate/atac/atac_counts_region_by_cell.tsv",
    "peak_metadata": "intermediate/atac/peak_metadata.bed",
    "motif_hits": "intermediate/atac/motif_hits.tsv",
    "region_to_gene": "intermediate/atac/region_to_gene_links.tsv",
    "regulators": "intermediate/metadata/regulator_list.txt",
    "targets": "intermediate/metadata/target_list.txt",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_input_contract(replicate_dir: str | Path) -> dict:
    replicate_dir = Path(replicate_dir)
    files = {}
    for name, relative in SHARED_OBSERVATIONS.items():
        path = replicate_dir / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing shared benchmark input: {path}")
        files[name] = {
            "relative_path": relative,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    return {
        "schema_version": 1,
        "replicate": replicate_dir.name,
        "files": files,
    }


def write_input_contract(replicate_dir: str | Path) -> Path:
    replicate_dir = Path(replicate_dir)
    contract = build_input_contract(replicate_dir)
    path = replicate_dir / "method_inputs" / "shared_input_contract.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    return path


def verify_input_contract(replicate_dir: str | Path) -> None:
    replicate_dir = Path(replicate_dir)
    path = replicate_dir / "method_inputs" / "shared_input_contract.json"
    expected = json.loads(path.read_text(encoding="utf-8"))
    current = build_input_contract(replicate_dir)
    if expected["files"] != current["files"]:
        raise RuntimeError(
            "Shared RNA/ATAC inputs changed after method preparation; "
            "regenerate all method inputs before benchmarking."
        )
