"""Export a historical SFT reference without rerunning simulation."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from rlinf_rft.common import ROOT, file_hash, load_manifest, select_seeds, write_json


def export_reference(campaign: Path, checkpoint: Path, output: Path) -> dict:
    campaign, checkpoint, output = (
        path.resolve() for path in (campaign, checkpoint, output)
    )
    repo = Path(os.environ.get("REPO_PATH", "/workspace/RLinf")).resolve()
    if any(output == tree or tree in output.parents for tree in (ROOT, repo)):
        raise ValueError("Historical reports must be outside synchronized source trees")
    if output.exists():
        raise FileExistsError(f"Refusing to replace an existing reference: {output}")

    source = campaign / "evaluation" / checkpoint.name
    manifest_path = campaign / "success_seeds.json"
    manifest = load_manifest(manifest_path, require_complete=True)
    selected = select_seeds(manifest, 8)
    seeds = selected["open_laptop"]["success_seeds"]
    run = json.loads((source / "run.json").read_text())
    summary = json.loads((source / "summary.json").read_text())
    runtime = json.loads((source / "model_runtime.json").read_text())
    config = yaml.safe_load((source / "config.yaml").read_text())
    records = [
        json.loads(line)
        for line in (source / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    recorded_seeds = [record["seed"] for record in records]
    expected = manifest["open_laptop"]
    if (
        not summary["complete"]
        or run["exit_code"] != 0
        or len(recorded_seeds) != len(set(recorded_seeds))
        or set(recorded_seeds) != set(expected["success_seeds"])
        or summary["total"] != len(records)
        or run["episodes"] != len(records)
        or summary["success"] != sum(record["success"] for record in records)
    ):
        raise ValueError("Historical run must have complete, unique episode coverage")
    if any(
        record["instruction"] != expected["instructions"][str(record["seed"])]
        for record in records
    ):
        raise ValueError("Historical instructions differ from the held-out manifest")
    if Path(config["rollout"]["model"]["model_path"]).resolve() != checkpoint:
        raise ValueError("Historical config points to another checkpoint")
    if runtime["seed"] != config["rollout"]["seed"]:
        raise ValueError("Historical rollout seed differs from the runtime report")
    for relative, digest in run["source_sha256"].items():
        if file_hash(ROOT / relative) != digest:
            raise ValueError(
                f"Task source changed since historical evaluation: {relative}"
            )

    provenance_path = campaign / "repro/checkpoints.json"
    provenance = json.loads(provenance_path.read_text())[checkpoint.name]
    model_hashes = {}
    for name in (
        "model.safetensors",
        "rlinf_model.yaml",
        "physical-intelligence/robotwin/norm_stats.json",
    ):
        print(f"Verifying checkpoint SHA256: {name}", flush=True)
        digest = file_hash(checkpoint / name)
        if digest != provenance[name]["sha256"]:
            raise ValueError(f"Checkpoint changed since historical evaluation: {name}")
        model_hashes[name] = digest

    by_seed = {record["seed"]: record for record in records}
    episodes = [by_seed[seed] for seed in seeds]
    inputs = [manifest_path, provenance_path] + [
        source / name
        for name in (
            "run.json",
            "summary.json",
            "model_runtime.json",
            "config.yaml",
            "seeds.json",
            "episodes.jsonl",
        )
    ]
    report = {
        "schema": "robotwin_historical_sft_reference/v1",
        "export_complete": True,
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "kind": "historical_reference",
        "strictly_paired": False,
        "limitations": [
            "These scenes were embedded in the original 100-episode execution order.",
            "Per-episode policy RNG states were not recorded; an independent eight-scene run cannot reproduce the original noise sequence.",
            "Use the eight-scene result for pipeline validation, not as evidence of an improvement in success rate.",
        ],
        "source_run": str(source),
        "checkpoint": str(checkpoint),
        "model_sha256": model_hashes,
        "source_sha256": {str(path): file_hash(path) for path in inputs},
        "original_evaluation_gpu": run["gpu"],
        "rollout_seed": runtime["seed"],
        "original_episode_indices_zero_based": {
            str(seed): recorded_seeds.index(seed) for seed in seeds
        },
        "full_evaluation": {
            "success": summary["success"],
            "total": summary["total"],
            "success_rate": summary["success_rate"],
        },
        "success": sum(record["success"] for record in episodes),
        "total": len(episodes),
        "success_rate": sum(record["success"] for record in episodes) / len(episodes),
        "episodes": episodes,
        "evaluation_seeds": str(output / "eval_seeds.json"),
        "recommended_post_training_gpu": run["gpu"],
        "ppo_gpu_selection": None,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "eval_seeds.json", selected)
    (output / "episodes.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in episodes)
    )
    for name in ("config.yaml", "run.json", "model_runtime.json", "summary.json"):
        shutil.copyfile(source / name, output / f"original_{name}")
    write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = export_reference(args.campaign, args.checkpoint, args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "success": report["success"],
                "total": report["total"],
                "strictly_paired": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
