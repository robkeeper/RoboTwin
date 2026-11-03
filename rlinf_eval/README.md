# 9981 open_laptop paired evaluation

## Parallel 100-episode campaign

`python -m rlinf_eval.campaign --output <bundle> --gpu-7272 3 --gpu-18180 6`
screens 50 expert-success scenes per GPU, using four independent processes
with quotas 13, 13, 12 and 12. Candidate ranges start at 100000/110000/120000/130000
on the first GPU and 200000/210000/220000/230000 on the second.
The merged 100-scene manifest fixes both seeds and
instructions for the two checkpoints. Each model uses one environment.

The campaign uses frozen source copies, copied task assets and SAPIEN, hashes
both checkpoints, records package versions, and writes persistent status,
child process metadata, console logs, per-episode results and GPU samples.
Each Ray instance has eight CPUs and a 1 GiB object store. A shared file lock
serializes CPU-heavy model initialization; rollouts run concurrently after
loading. `model_runtime.json` records actual precision and Torch/Ray settings.

The bundle's `reproduce.sh GPU_7272 GPU_18180 [new_output]` reuses the frozen
100 scenes and archived runtime, and rejects checkpoints whose hashes differ
from the reference. `report.json`, `paired_results.json` and
`artifact_checksums.json` are written after both full evaluations finish.

Code is edited in the local RoboTwin fork and synchronized by the
`robotwin-code-9981` Mutagen session to `/data0/wangyihan/RoboTwin`.
The container `wangyihan_robotwin_rlinf` sees it at `/workspace/RoboTwin`.
Git metadata is kept locally; runtime output is outside the synchronized tree.
The synchronization exclusions retain the tracked `data/decode_image_bit.py`
and `data/process_stuck.py` utilities while ignoring generated data.

## Runtime

- Python: `/opt/venv/openpi/bin/python`
- RLinf: `/workspace/RLinf`
- Assets: `/data/robotwin_assets`, selected with `ROBOTWIN_ASSETS_PATH`
- Reference SAPIEN 3.0.0b1: `/workspace/ctr_backup/sapien300b1`
- Checkpoints: `/data/robotwin_ckpt/rlinf/{7272,18180}`
- Output: `/workspace/ctr_backup/open_laptop_eval`

The base image needed the packages pinned in `requirements-extra.txt`.
Set `RAY_ADDRESS=local` for evaluation: this host runs other Ray clusters,
and automatic discovery can select an instance outside the container.

Reserve a GPU and confirm its index before running these commands. Initial
estimates for one environment were 2–6 GiB for expert screening and 22–30 GiB
for model plus simulation. The completed GPU 6 smoke run sampled a peak of
20.43 GiB across the paired runs. Allow at least 32 GiB free on one GPU for
headroom; sampling does not capture every instantaneous allocation peak.
The two checkpoints run sequentially and do not need two GPUs.

## Verified smoke results, 2026-11-03

All four boundary tests passed. Expert screening accepted seeds 100002,
100003, 100004 and 100006 from seven candidates. Both checkpoint runs exited
zero and each covered all four seeds once with identical instructions.

| Checkpoint | Success | Sampled GPU peak |
|---|---:|---:|
| 7272 | 2/4 (50%) | 20.43 GiB |
| 18180 | 3/4 (75%) | 17.21 GiB |

These four-episode results verify the pipeline and are too small to establish
the final benchmark success rate. The separate initial one-episode pilot is
not included in this table.

Server results: `/data0/wangyihan/ctr_backup/open_laptop_eval/20261103_171819_332966`.
Verified local copies: `results/20261103_gpu6_smoke/`, including a compact
`test_report.json`, full configs, per-seed results, console logs and GPU samples.

## Protocol

`open_laptop.yaml` fixes clean scenes, aloha-agilex, three RGB cameras, mplib,
native `take_action`, 700 environment actions, one environment and 50-action
chunks. Each checkpoint's own `rlinf_model.yaml` supplies its model settings,
including FP32 inference, 10 denoising steps and normalization transforms.
The rollout worker starts from the same explicit RNG seed after model loading.

The new `robotwin.envs.vector_env` adapter uses fork/main's existing task API.
`take_action` is the default. `chunk` is accepted only for a task that provides
`gen_sparse_reward_data`; fork/main's open_laptop does not provide that method.
The mplib backend is optional; existing native task calls still default to
curobo when no `planner_backend` is supplied.

Expert screening starts at seed 100000 by default, runs `play_once()`, and
accepts only `plan_success and check_success()`. It saves the exact instruction
with each seed. Unstable scenes are recorded and skipped; unexpected errors
stop screening. Incomplete screening can resume with the same arguments.
Completed manifests are reused without running the experts again. Evaluation
setup errors never silently substitute another seed.

## After the GPU reservation is confirmed

Replace `G` below with the reserved physical GPU index. The launcher requires
exactly one GPU index and sets both CUDA visibility and RLinf placement.
The first runtime check must confirm the SAPIEN renderer and rollout process
are on the reserved card. `ROBOTWIN_RENDER_DEVICE` defaults to `cuda:0` within
the restricted CUDA visibility and can be overridden if driver mapping differs.

First run the CPU boundary checks, after authorization to start testing:

```bash
docker exec -e CUDA_VISIBLE_DEVICES=G \
  -e ROBOTWIN_ASSETS_PATH=/data/robotwin_assets \
  -e VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
  -e PYTHONPATH=/workspace/ctr_backup/sapien300b1:/workspace/RoboTwin \
  -w /workspace/RoboTwin wangyihan_robotwin_rlinf \
  /opt/venv/openpi/bin/python -m unittest discover \
  -s tests -p 'test_*adapter.py'
```

Screen four expert-success scenes for the smoke run:

```bash
docker exec -e GPU_LIST=G -e RAY_ADDRESS=local wangyihan_robotwin_rlinf \
  bash /workspace/RoboTwin/rlinf_eval/run.sh scan --count 4 \
  --output /workspace/ctr_backup/open_laptop_eval/smoke_seeds.json
```

Run one episode first to measure memory and inspect device placement:

```bash
docker exec -e GPU_LIST=G -e RAY_ADDRESS=local wangyihan_robotwin_rlinf \
  bash /workspace/RoboTwin/rlinf_eval/run.sh run --checkpoint 7272 --episodes 1 \
  --seeds /workspace/ctr_backup/open_laptop_eval/smoke_seeds.json
```

Then run four episodes per checkpoint, sequentially:

```bash
docker exec -e GPU_LIST=G -e RAY_ADDRESS=local wangyihan_robotwin_rlinf \
  bash /workspace/RoboTwin/rlinf_eval/run.sh run --checkpoint both --episodes 4 \
  --seeds /workspace/ctr_backup/open_laptop_eval/smoke_seeds.json
```

For the final comparison, screen 100 scenes into a separate manifest and run
both checkpoints with `--episodes 100` using that manifest. Results from the
old demonstration `seed.txt` are not used as a held-out evaluation set.

Each run creates a new timestamp directory with separate checkpoint folders.
Each folder contains `config.yaml`, `seeds.json`, `run.json`, `console.log`,
`episodes.jsonl` and `summary.json`. A final success rate is valid only when
the process exits zero and every requested seed has exactly one result.
`gpu_usage.jsonl` records GPU memory samples; `comparison.json` collects both summaries. Unexpected scene failures stop the
run and do not count as policy failures.

Tests are in `tests/test_rlinf_adapter.py` and `tests/test_mplib_adapter.py`. They cover stopping on
success, action-budget enforcement inside a chunk, observation padding, and
preservation of a requested seed when setup fails, and full-robot joint-state
forwarding to mplib. All four tests passed on 2026-11-03.
