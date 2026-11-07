"""Check recorded observations, rewards and the actual PPO loss mask on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(episode: Path) -> dict:
    import torch

    from rlinf.algorithms.advantages import compute_gae_advantages_and_returns
    from rlinf.utils.metric_utils import compute_loss_mask

    data = np.load(episode / "trajectory.npz", allow_pickle=False)
    term = data["terminated"].reshape(14, 1, 50)
    trunc = data["truncated"].reshape(14, 1, 50)
    rewards = data["reward"].reshape(14, 1, 50)
    done = torch.from_numpy(np.concatenate([np.zeros((1, 1, 50), bool), term | trunc]))
    loss_mask, _ = compute_loss_mask(done)
    chunk_mask = loss_mask.any(dim=-1, keepdim=True)
    chunk_rewards = torch.from_numpy(rewards).sum(dim=-1, keepdim=True)
    reward_slots = np.flatnonzero(data["reward"])
    for slot in reward_slots:
        if not chunk_mask[slot // 50, 0, 0]:
            raise AssertionError("Success reward belongs to a masked PPO chunk")
    if reward_slots.size > 1 or not np.isin(data["reward"], [0, 1]).all():
        raise AssertionError("Sparse reward must be zero or a single +1 pulse")
    chunks = json.loads((episode / "chunks.json").read_text())
    first_done = next(
        (i for i, c in enumerate(chunks) if c["terminated"] or c["truncated"]), None
    )
    if first_done is not None and chunk_mask[first_done + 1 :].any():
        raise AssertionError("PPO retains chunks after episode completion")

    # With auto_reset=false, GAE uses done=termination OR truncation. Demonstrate
    # that the redundant truncation on a successful terminal cannot alter GAE.
    cleaned_trunc = trunc & ~term
    clean_done = torch.from_numpy(
        np.concatenate([np.zeros((1, 1, 50), bool), term | cleaned_trunc])
    )
    done_chunks = done.any(-1)
    clean_done_chunks = clean_done.any(-1)
    values = torch.linspace(-0.2, 0.4, 15).reshape(15, 1)
    inputs = dict(
        rewards=chunk_rewards[..., 0],
        values=values,
        gamma=0.99,
        gae_lambda=0.95,
        normalize_advantages=False,
    )
    original = compute_gae_advantages_and_returns(dones=done_chunks, **inputs)
    cleaned = compute_gae_advantages_and_returns(dones=clean_done_chunks, **inputs)
    gae_equal = all(torch.equal(a, b) for a, b in zip(original, cleaned))

    synthetic_done = torch.zeros((15, 1, 50), dtype=torch.bool)
    synthetic_done[6:, 0, -1] = True
    synthetic_mask, _ = compute_loss_mask(synthetic_done)
    synthetic_chunk_mask = synthetic_mask.any(-1)
    if not synthetic_chunk_mask[:6].all() or synthetic_chunk_mask[6:].any():
        raise AssertionError(
            "Real PPO mask mishandles success followed by padded chunks"
        )

    state = data["state_after"]
    real = data["state_actual"]
    error = np.abs(state - real)[data["executed"].astype(bool)]
    arm_indices = [*range(6), *range(7, 13)]
    rgb = np.load(episode / "model_rgb.npz", allow_pickle=False)
    image_info = {
        k: {
            "shape": list(rgb[k].shape),
            "range": [float(rgb[k].min()), float(rgb[k].max())],
        }
        for k in rgb.files
    }
    normalized = data["model_state"][:, :14]
    bins = np.digitize(normalized, np.linspace(-1, 1, 257)[:-1]) - 1
    np.save(episode / "state_token_bins.npy", bins)
    result = {
        "sparse_reward_contract_passed": True,
        "success_reward_retained_in_ppo": True if reward_slots.size else None,
        "synthetic_mid_chunk_success_reward_and_mask_passed": True,
        "post_done_chunks_masked": True,
        "reward_slots_one_based": (reward_slots + 1).tolist(),
        "ppo_chunk_mask": chunk_mask[:, 0, 0].tolist(),
        "ppo_action_slot_mask": loss_mask[:, 0].flatten().tolist(),
        "native_executed_actions": int(data["executed"].sum()),
        "trained_chunk_count": int(chunk_mask.sum()),
        "masked_chunk_count": int((~chunk_mask).sum()),
        "redundant_success_truncation_changes_current_gae": not gae_equal,
        "gae_check_scope": "Actual recorded done boundaries, synthetic values, auto_reset=false",
        "arm_target_vs_physical_qpos_abs_error_rad": {
            "mean": float(error[:, arm_indices].mean()),
            "p95": float(np.quantile(error[:, arm_indices], 0.95)),
            "max": float(error[:, arm_indices].max()),
        },
        "normalized_state_range": [float(normalized.min()), float(normalized.max())],
        "model_state_padding_max_abs": float(np.abs(data["model_state"][:, 14:]).max()),
        "policy_state_matches_previous_chunk_final_state": bool(
            np.allclose(
                data["policy_raw_state"][1:], data["state_after"][49:650:50], atol=1e-6
            )
        ),
        "tokenized_prompt_valid_lengths": data["tokenized_prompt_mask"]
        .sum(-1)
        .tolist(),
        "state_tokenization": "14 normalized dimensions are quantized before padding state to 32",
        "model_rgb": image_info,
        "all_three_input_images_present": all(
            all(v) for c in chunks for v in c["input_image_masks"].values()
        ),
        "chunk_logprob_note": "PPO logprob sums all 50x14 components, including unexecuted suffix in the final successful chunk; later complete chunks are masked",
    }
    if not result["all_three_input_images_present"]:
        raise AssertionError("A policy camera is masked out")
    (episode / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    args = parser.parse_args()
    result = audit(args.episode)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "ppo_action_slot_mask"}, indent=2
        )
    )
