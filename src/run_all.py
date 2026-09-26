from __future__ import annotations

import json
import subprocess

from . import config
from .beir_suite import run_beir_suite
from .experiments import run_baseline, run_quantization, run_scoring
from .index import index_scifact_deep_dive
from .qdrant_utils import wait_ready
from .ranking import run_ranking
from .adaptive_pipeline import run_adaptive_pipeline
from .fusion_lab import run_fusion_lab
from .router import run_router


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return ""


def snapshot_hardware() -> dict:
    mem = _cmd(["free", "-h"])
    gpu = _cmd(["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"])
    docker = _cmd(["docker", "stats", "qdrant_local", "--no-stream", "--format", "{{.MemUsage}}"])
    disk = _cmd(["du", "-sh", str(config.ROOT / "qdrant_storage")])
    cpu = _cmd(["nproc"])
    return {
        "cpu_threads": cpu,
        "memory_free_h": mem.splitlines()[1] if mem else "",
        "gpu": gpu,
        "qdrant_container_mem": docker,
        "qdrant_storage_du": disk,
    }


def _qdrant_version() -> str:
    try:
        import urllib.request

        with urllib.request.urlopen(config.QDRANT_URL) as resp:
            return json.loads(resp.read().decode()).get("version", "unknown")
    except Exception:
        return "1.18.1"


def main() -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Qdrant health ===")
    wait_ready()
    version = _qdrant_version()
    hw = snapshot_hardware()
    print(hw, version)
    (config.RESULTS_DIR / "hardware.json").write_text(json.dumps(hw, indent=2))

    print("=== Full BEIR suite ===")
    suite = run_beir_suite()
    hw["after_suite"] = snapshot_hardware()
    (config.RESULTS_DIR / "hardware.json").write_text(json.dumps(hw, indent=2))

    print("=== SciFact deep dive collections ===")
    index_meta = index_scifact_deep_dive()
    index_meta["qdrant_version"] = version
    index_meta["beir_evaluated"] = suite.get("n_evaluated")
    index_meta["beir_skipped"] = suite.get("n_skipped")
    (config.RESULTS_DIR / "index.json").write_text(json.dumps(index_meta, indent=2, default=str))

    print("=== Scoring / quantization / router / ranking / baseline (SciFact) ===")
    run_scoring()
    run_quantization()
    run_router()
    run_fusion_lab()
    run_adaptive_pipeline()
    run_ranking()
    run_baseline()
    hw["after_deep_dive"] = snapshot_hardware()
    (config.RESULTS_DIR / "hardware.json").write_text(json.dumps(hw, indent=2))


if __name__ == "__main__":
    main()
