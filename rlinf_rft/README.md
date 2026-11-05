# Open-laptop Pi0.5 PPO validation

This package uses RLinf's stock PPO runners and success-termination reward.
All source edits belong in the local checkouts and reach 9981 through Mutagen.
Run artifacts must be outside both synchronized trees.

The initialization is `/data/robotwin_ckpt/rlinf/7272`. Preserve FP32, T50,
10 denoising steps, the checkpoint's norm stats, `adapt_to_pi=false`, native
`take_action`, clean ALOHA scenes, SAPIEN 3.0.0b1 and the 700-action budget.
Success is termination; timeout is truncation and never earns success reward.
RLinf's relative custom reward gives one +1 pulse per successful episode.

The three task YAMLs live in RLinf. Python entry points, boundary tests,
resource monitoring and checkpoint comparisons live here. Worker subclasses
add loading locks, seeds and reports; PPO/FSDP calculations remain upstream.
The actor model-loading lock is released before any FSDP collective.

## CPU checks (no GPU launch)

```bash
bash /workspace/RoboTwin/rlinf_rft/run.sh check-config \
  --gpus 0,3,6 --checkpoint /data/robotwin_ckpt/rlinf/7272

CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/workspace/RoboTwin:/workspace/RLinf \
  /opt/venv/openpi/bin/python -B -m unittest discover \
  -s /workspace/RoboTwin/tests -p 'test_rlinf*.py'
```

`resources` only reads nvidia-smi and `/proc/meminfo`. After a PPO probe,
`resources --peak-mib M` admits cards with at least M+4096 MiB free, including
cards already running other processes. Review utilization before choosing at
most six physical GPUs. Freeze the choice for each formal run.

## Prepare scenes and configuration

After confirming the screening GPU, run `GPU_LIST=G bash rlinf_rft/run.sh scan
--output /workspace/open_laptop_ppo/scenes/train56.json`. Defaults are start
300000, 56 accepted expert scenes and at most 1000 candidates. Screening saves
instructions and can resume an incomplete manifest with identical settings.

To run without keeping SSH or an assistant active, use the background supervisor
from `/workspace/RoboTwin`:

```bash
/opt/venv/openpi/bin/python -B -m rlinf_rft.background launch \
  --gpu 5 --start 300000 --count 56 --max-attempts 1000 \
  --output /workspace/open_laptop_ppo/scenes/train56.json
```

It detaches from the terminal, retains a job lock to reject duplicate launches,
resumes the existing manifest, and stops at 56 accepted scenes. Reaching the
1000-candidate cap without enough scenes also stops and records failure.
`train56.log` preserves stdout/stderr and `train56.job.json` records the worker
PID, final exit code and completion state. The original manifest is updated
after each candidate. `background status --output PATH` reads current progress
once; no agent or periodic AI monitoring is required. An attached old scan can
be handed over with `launch --replace-pid PID`; this only stops the exact scan
module with the matching output path and restarts any unfinished candidate.

The existing 100-scene evaluation manifest is at
`/workspace/ctr_backup/open_laptop_eval/full100_20261103_gpu3_6/success_seeds.json`.
The demonstration seed file is
`/data/RoboTwin/dataset/open_laptop/aloha-agilex_clean_50/seed.txt`.

```bash
bash rlinf_rft/run.sh prepare --gpus 0,3,6 \
  --train-seeds /workspace/open_laptop_ppo/scenes/train56.json \
  --eval-seeds /workspace/ctr_backup/open_laptop_eval/full100_20261103_gpu3_6/success_seeds.json \
  --demo-seeds /data/RoboTwin/dataset/open_laptop/aloha-agilex_clean_50/seed.txt \
  --output /workspace/open_laptop_ppo/RUN_ID
```

Preparation rejects overlapping seeds, incomplete screening, missing
instructions and output inside a synchronized source tree. Training uses 2N
environments, micro batch 1, global batch 4N and a seed subset divisible by 2N.
The remainder is counted in run.json. All eight held-out scenes and their
instructions stay fixed; before/after evaluation uses the first chosen GPU.

## GPU stages (confirm each stage before executing)

The existing 7272 evaluation can also be exported as a historical reference on
CPU, without launching another baseline evaluation:

```bash
CUDA_VISIBLE_DEVICES= /opt/venv/openpi/bin/python -B -m rlinf_rft.reuse_baseline \
  --campaign /workspace/ctr_backup/open_laptop_eval/full100_20261103_gpu3_6 \
  --checkpoint /data/robotwin_ckpt/rlinf/7272 \
  --output /workspace/open_laptop_ppo/baseline_7272_historical_20261104
```

This checks complete episode coverage, fixed instructions, task source hashes,
checkpoint weights and norm stats against the original evaluation provenance.
The first eight scenes sorted by seed had five successes. The exported report
marks them as a historical reference: their original execution order and policy
noise sequence differ from an independent eight-scene run. This replaces the
new pre-training evaluation only when the user accepts that limitation. It does
not freeze the PPO GPU selection. The original evaluation GPU was 3; retain it
for post-training evaluation when resources permit. Keep the historical report
alongside the final training run and compare its exact exported scene manifest.

For the example above, set GPU_LIST to `0` for the eight-scene baseline:
`bash rlinf_rft/run.sh eval --run-dir /workspace/open_laptop_ppo/RUN_ID --label before`.

Set GPU_LIST to `0,3,6` for `train --run-dir ... --probe`. The probe runs one
outer iteration with zero learning rates, including the optimizer allocation
and the configured inner updates. It is separate from formal training.
Read `probe/resource_summary.json` and the per-worker allocator peaks; process
samples are not an upper bound on transient peaks. Allow at least 4 GiB more.

After resource review and confirmation, run `train --run-dir ... --peak-mib M`.
It rejects GPUs without the required current headroom and runs exactly three
formal PPO iterations, saving each checkpoint. A run directory is never reused
for another training process. Each stage uses a private Ray instance and short
temporary directory, without stopping unrelated Ray instances.

Run `verify --run-dir ...` on CPU to compare all backbone parameters against
SFT: at least one trainable backbone tensor and value-head tensor must change,
and every frozen tensor must remain equal. It checks shapes, finite weights,
three checkpoint files, three metric reports and a nonzero successful-episode
count. Then evaluate the `global_step_3` directory using `eval --label after
--checkpoint PATH` on the same physical GPU as before. This also verifies that
the saved RLinf checkpoint can be loaded for inference.

Keep model/source fingerprints, manifests, stdout, TensorBoard, per-episode
records and resource reports. Three PPO iterations validate the pipeline; they
do not establish that success rate has improved. Commit both local repos only
after the agreed validation, then push only the RoboTwin branch to its origin.

## Terminal status and conditional long training

From the local project root, the user can run the standard-library-only watcher:

```bash
python3 watch_open_laptop_ppo.py
# Exit after the complete-model probe passes:
python3 watch_open_laptop_ppo.py --until probe
```

The project-root wrapper automatically uses the local `codex` Conda interpreter
at `/opt/anaconda3/envs/codex/bin/python` and prints the active prefix. The watcher
uses only Python's standard library, so no additional pip packages are required.
Its versioned entry point is `RoboTwin/watch_open_laptop_ppo.py`; the local
workspace-root entry point is a symbolic link to that file.

It reads `9981:/data0/wangyihan/open_laptop_ppo/auto_7272/status.json` once per
minute, prints each stage event once, and retries temporary connection errors.
`--status PATH` selects another server job; `--once` reads a single snapshot.
Ctrl+C stops the local watcher. It does not stop remote training or invoke AI.

After the user confirms the training configuration and a low-memory smoke run
has passed, the server workflow can run independently of this watcher:

```bash
bash /workspace/RoboTwin/rlinf_rft/run.sh workflow \
  --smoke-report /workspace/open_laptop_ppo/smoke/report.json
```

The low-memory test is `GPU_LIST=5 bash rlinf_rft/run.sh smoke --output
/workspace/open_laptop_ppo/smoke`. It exercises one native three-camera
environment, T50 hold actions/reset, boundary tests, RLinf's PPO loss/backward
with a tiny policy, and checkpoint/TensorBoard roundtrips. The default total
sampled process budget is 4 GiB, with a 2 GiB PyTorch allocator limit. It does not
load the Pi0.5 model. `python -m rlinf_rft.smoke --device cpu --output PATH` checks
the CPU plumbing only; its report cannot authorize the GPU workflow.

The workflow waits for the same three cards to meet 44 GiB free and at most 10%
utilization on three checks 60 seconds apart, with 64 GiB host memory and 1 TiB
disk free. It freezes the selected GPUs, probes the complete model for one round
at zero learning rates, and requires its conservative measured peak plus 4 GiB
before starting the three-round validation. It uses gradient checkpointing with
`use_reentrant=false`, six environments, micro batch 1 and global batch 12.

Long training requires finite loss/gradient/KL metrics, changed action/value
parameters, an unchanged frozen backbone, the full/sharded checkpoint, recorded
reward/return TensorBoard metrics, and successful execution of all eight fixed
reload evaluation scenes on the original GPU 3. Success rate improvement is not
a gate. The workflow then waits for the same training GPUs if needed and starts
an **independent 300-round run from the original 7272 SFT weights**. Its optimizer
and scheduler are newly created, `resume_dir` is null, and its counter starts at
zero. Validation checkpoints are saved every round; formal checkpoints are saved
at steps `10, 20, ..., 300`, ending at `global_step_300` in the separate
`long_train` directory. The initialization manifest records that the validated
step-3 weights/optimizer are not used for this fresh run. Stage logs, event
history, resources and both resolved configs
remain under `/workspace/open_laptop_ppo/auto_7272`.

The watcher can start before the server job exists; it prints `not_started` and
waits. Writing these scripts does not launch the workflow. The smoke test and
GPU workflow still require the agreed configuration confirmation.

For an existing probe or its following resource check, the project-root
`resume_open_laptop_ppo.py --job-dir JOB_DIR` replaces the controller while
preserving the running probe process. It checks resources once and retries only
while the capacity requirements are unmet. `--training-gpus 3,4,5` can select a
same-size placement before formal training. This derives a separate resolved
run with identical model, hyperparameters and frozen seed manifests; the probe
and its original source fingerprints remain available. The controller hash and
policy change are recorded in `plan.json`. The active run uses the approved 30%
utilization limit and GPU 3/4/5 for validation and formal training.
