#!/usr/bin/env python3
"""Smoke test for OPD process reward with two different continuation models.

This script mirrors the process-reward computation in
`verl/verl/trainer/ppo/opd_consistency.py` (the same logic used by
`on_policy_distillation.sh`) but replaces the distributed verl rollout worker
with a local vLLM `LLM` instance.

For efficiency, the main reasoning trajectory is generated **only once** and
split into segments.  Both configured continuation models then evaluate the
same set of segment prefixes, so the comparison is over the same trajectory
rather than two independently generated trajectories.

Pipeline:

    1. Generate one main response trajectory per prompt (using the model named
       in `MAIN_RESPONSE_MODEL`).
    2. Split the response into segments at the OPD trigger words.
    3. For each segment boundary (except the final one), build a prefix prompt
       and generate K suffix continuations with each continuation model.
    4. Score each continuation with the math reward function used by the shell
       script.
    5. Convert endpoint accuracies into process rewards and merge adjacent
       segments with the same sign, exactly as the original code does.

Run inside the `verl` conda environment:

    conda activate verl
    python smoke_test_process_reward.py

Output: `smoke_test_pr_results.parquet` with the original prompt/ground truth,
the shared generated response, and the merged segment bounds / process rewards
for both continuation models.
"""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# Make sure the local `verl` package is importable (the same one used by the
# training script).  We need the reward function defined under
# verl/verl/utils/reward_score/ttrl_math/__init__.py.
OPD_ROOT = Path(__file__).resolve().parent
if str(OPD_ROOT) not in sys.path:
    sys.path.insert(0, str(OPD_ROOT))

from verl.utils.reward_score.ttrl_math import reward_func as raw_reward_func  # noqa: E402

# ---------------------------------------------------------------------------
# Constants copied from verl/verl/trainer/ppo/opd_consistency.py
# ---------------------------------------------------------------------------

_SEGMENT_TRIGGER_RE = re.compile(
    r"\bwait\b|\balternatively\b|\blet\s+me\s+check\b|"
    r"\blet\s+me\s+reconsider\b|\bhold\s+on\b|\bactually\b|\bhmm\b",
    re.IGNORECASE,
)

EPISODE_EVAL_SUFFIX = (
    "\nBased on the reasoning above, directly give the final answer. "
    "Put the final answer within \\boxed{}. "
    "If no definitive answer can be derived from the existing reasoning, "
    "output \\boxed{no answer}."
)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return default


# ---------------------------------------------------------------------------
# Process reward configuration (matches the environment variables used in
# on_policy_distillation.sh)
# ---------------------------------------------------------------------------

@dataclass
class ProcessRewardConfig:
    enable: bool = True
    k_eval: int = 8
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    max_tokens: int = 300
    max_prompt_length: int = 8192
    max_total_length: int = 8192
    batch_size: int = 64
    stop: tuple[str, ...] = ()


@dataclass
class ConsistencyConfig:
    enable: bool = True
    top_percent_responses: float = 30.0
    mask_top_percent_segments: float = 20.0
    min_segments_per_response: int = 3
    min_sentences_per_segment: int = 3
    drop_low_consistency: bool = True
    strict_pr_required: bool = True
    max_segments: int = 128
    process_reward: ProcessRewardConfig = field(default_factory=ProcessRewardConfig)


def env_process_reward_config() -> ProcessRewardConfig:
    """Read the OPD process-reward settings from environment variables."""
    return ProcessRewardConfig(
        enable=_as_bool(os.environ.get("OPD_PROCESS_REWARD_ENABLE", "True")),
        k_eval=int(os.environ.get("OPD_PROCESS_REWARD_K_EVAL", "8")),
        temperature=float(os.environ.get("OPD_PROCESS_REWARD_TEMPERATURE", "0.7")),
        top_k=int(os.environ.get("OPD_PROCESS_REWARD_TOP_K", "50")),
        top_p=float(os.environ.get("OPD_PROCESS_REWARD_TOP_P", "1.0")),
        max_tokens=int(os.environ.get("OPD_PROCESS_REWARD_MAX_TOKENS", "300")),
        max_prompt_length=int(os.environ.get("OPD_PROCESS_REWARD_MAX_PROMPT_LENGTH", "8192")),
        max_total_length=int(os.environ.get("OPD_PROCESS_REWARD_MAX_TOTAL_LENGTH", "8192")),
        batch_size=int(os.environ.get("OPD_PROCESS_REWARD_BATCH_SIZE", "64")),
    )


def env_consistency_config() -> ConsistencyConfig:
    pr_cfg = env_process_reward_config()
    return ConsistencyConfig(
        enable=_as_bool(os.environ.get("OPD_CONSISTENCY_ENABLE", "True")),
        top_percent_responses=float(os.environ.get("OPD_CONSISTENCY_TOP_PERCENT_RESPONSES", "70")),
        mask_top_percent_segments=float(os.environ.get("OPD_CONSISTENCY_MASK_TOP_PERCENT_SEGMENTS", "70")),
        min_segments_per_response=int(os.environ.get("OPD_CONSISTENCY_MIN_SEGMENTS", "3")),
        min_sentences_per_segment=int(os.environ.get("OPD_SEGMENT_MIN_SENTENCES", "3")),
        drop_low_consistency=_as_bool(os.environ.get("OPD_CONSISTENCY_DROP_LOW", "True"), True),
        strict_pr_required=_as_bool(os.environ.get("OPD_CONSISTENCY_STRICT_PR_REQUIRED", "True"), True),
        max_segments=int(os.environ.get("OPD_CONSISTENCY_MAX_SEGMENTS", "128")),
        process_reward=pr_cfg,
    )

def _sentence_count(text: str) -> int:
    return text.count(".")


def _segment_text(text: str, max_segments: int, min_sentences_per_segment: int = 5) -> list[tuple[int, int]]:
    """Split reasoning at OPD trigger words, avoiding short segments."""
    if not text.strip():
        return []

    min_sentences = max(1, int(min_sentences_per_segment))
    spans: list[tuple[int, int]] = []
    pos = 0

    for match in _SEGMENT_TRIGGER_RE.finditer(text):
        end = match.start()
        if end <= pos or not text[pos:end].strip():
            continue
        if _sentence_count(text[pos:end]) >= min_sentences and len(spans) < max_segments - 1:
            spans.append((pos, end))
            pos = end

    if pos < len(text):
        spans.append((pos, len(text)))

    if not spans:
        spans = [(0, len(text))]
    return spans[:max_segments]


def _merge_same_sign_segments(
    bounds: list[tuple[int, int]],
    process_rewards: list[float],
) -> tuple[list[tuple[int, int]], list[float]]:
    """Merge adjacent segments while the process-reward sign does not flip."""
    n = min(len(bounds), len(process_rewards))
    if n <= 1:
        return bounds[:n], process_rewards[:n]

    merged_bounds: list[tuple[int, int]] = []
    merged_prs: list[float] = []
    run_start, run_end = bounds[0]
    run_pr = float(process_rewards[0])
    run_sign = 1 if run_pr > 0 else (-1 if run_pr < 0 else 0)

    for span, pr_val in zip(bounds[1:n], process_rewards[1:n]):
        pr = float(pr_val)
        sign = 1 if pr > 0 else (-1 if pr < 0 else 0)
        if sign == 0:
            run_end = span[1]
            continue
        sign_flip = run_sign != 0 and sign != run_sign
        if sign_flip:
            merged_bounds.append((run_start, run_end))
            merged_prs.append(run_pr)
            run_start, run_end = span
            run_pr = pr
            run_sign = sign
            continue

        run_end = span[1]
        run_pr += pr
        if run_sign == 0:
            run_sign = sign

    merged_bounds.append((run_start, run_end))
    merged_prs.append(run_pr)
    return merged_bounds, merged_prs


# ---------------------------------------------------------------------------
# Reward wrapper (same call signature used by the training code)
# ---------------------------------------------------------------------------

def _score_completion(data_source: Any, text: str, ground_truth: Any) -> float:
    if ground_truth is None:
        return 0.0
    score = raw_reward_func(data_source=data_source, solution_str=text, ground_truth=ground_truth, extra_info={})
    if isinstance(score, dict):
        return float(score.get("score", 0.0))
    if isinstance(score, (tuple, list)):
        return float(score[0]) if score else 0.0
    return float(score)


def _answer_string_present(text: str, ground_truth: Any) -> bool:
    if ground_truth is None:
        return False
    if isinstance(ground_truth, np.ndarray):
        ground_truth = ground_truth.tolist()
    if isinstance(ground_truth, (list, tuple, set)):
        return any(_answer_string_present(text, item) for item in ground_truth)
    answer = str(ground_truth).strip()
    if not answer:
        return False
    return answer in text


# ---------------------------------------------------------------------------
# vLLM continuation engine (replaces the distributed actor_rollout_wg in the
# original pipeline)
# ---------------------------------------------------------------------------

class VLLMContinuationEngine:
    """Local vLLM wrapper used as the continuation model for process rewards."""

    def __init__(self, model_path: str, gpu_memory_utilization: float = 0.8, dtype: str = "bfloat16"):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.llm = LLM(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype=dtype,
            gpu_memory_utilization=gpu_memory_utilization,
            tensor_parallel_size=1,
            max_model_len=8192,
        )
        self.model_name = Path(model_path).name

    def _build_prompt_text(self, raw_prompt: list[dict[str, Any]]) -> str:
        return self.tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)

    def _build_prefix_prompt(self, raw_prompt: list[dict[str, Any]], response_prefix: str) -> str:
        return self._build_prompt_text(raw_prompt) + (response_prefix or "") + EPISODE_EVAL_SUFFIX

    def _truncate_prompts(self, prompt_texts: list[str], max_prompt_length: int) -> list[str]:
        """Truncate prompt texts to the configured token budget."""
        if not prompt_texts:
            return []
        enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
        )
        truncated: list[str] = []
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"]):
            valid_ids = ids[mask.bool()].tolist()
            truncated.append(self.tokenizer.decode(valid_ids, skip_special_tokens=False))
        return truncated

    def _prompt_lengths(self, prompt_texts: list[str]) -> list[int]:
        """Return actual token lengths (excluding padding)."""
        if not prompt_texts:
            return []
        enc = self.tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        return enc["attention_mask"].sum(dim=-1).tolist()

    def generate(
        self,
        prompt_texts: list[str],
        pr_cfg: ProcessRewardConfig,
        max_tokens_list: list[int] | None = None,
        n: int = 1,
    ) -> list[str]:
        """Generate `n` continuations for each prompt text.

        `max_tokens_list` can be used to set a per-prompt max-token budget, which
        is required when the prefix prompts have different lengths and we must
        keep prompt + max_tokens within the total context budget.
        """
        if not prompt_texts:
            return []

        if max_tokens_list is None:
            max_tokens_list = [pr_cfg.max_tokens] * len(prompt_texts)
        if len(max_tokens_list) != len(prompt_texts):
            raise ValueError("max_tokens_list length must match prompt_texts length")

        # vLLM supports a list of SamplingParams paired one-to-one with prompts.
        sampling_params_list = [
            SamplingParams(
                n=n,
                temperature=pr_cfg.temperature,
                top_k=pr_cfg.top_k,
                top_p=pr_cfg.top_p,
                max_tokens=max(1, int(mt)),
                stop=list(pr_cfg.stop),
            )
            for mt in max_tokens_list
        ]

        texts: list[str] = []
        batch_size = max(1, pr_cfg.batch_size)
        for start in range(0, len(prompt_texts), batch_size):
            chunk = prompt_texts[start : start + batch_size]
            chunk_params = sampling_params_list[start : start + batch_size]
            outputs = self.llm.generate(chunk, chunk_params, use_tqdm=True)
            for out in outputs:
                for generated in out.outputs:
                    texts.append(generated.text)
        return texts

    def generate_main_responses(
        self,
        items: list[dict[str, Any]],
        pr_cfg: ProcessRewardConfig,
        cfg: ConsistencyConfig,
        main_max_tokens: int = 4096,
    ) -> tuple[list[dict[str, Any]], list[list[dict[str, Any]]]]:
        """Generate one reasoning trajectory per item and split it into segments.

        Returns the per-item records and the chat-format raw prompts.  The
        records are shared across all continuation models, while the raw prompts
        are used by each continuation model to build its own prefix prompts.
        """
        raw_prompts = [[{"role": "user", "content": item["prompt"]}] for item in items]
        main_prompt_texts = [self._build_prompt_text(p) for p in raw_prompts]
        main_prompt_texts = self._truncate_prompts(main_prompt_texts, pr_cfg.max_prompt_length)
        main_texts = self.generate(
            main_prompt_texts,
            pr_cfg,
            max_tokens_list=[main_max_tokens] * len(main_prompt_texts),
            n=1,
        )

        records = []
        for i, item in enumerate(items):
            response_text = main_texts[i]
            bounds = _segment_text(response_text, cfg.max_segments, cfg.min_sentences_per_segment)
            bounds = bounds[: cfg.max_segments]
            ground_truth = item.get("ground_truth")
            # `answer_present` is a cheap substring precheck used only for the
            # skip_pr gate below (skip when the GT number never appears at all
            # -- those trajectories are hopeless and not worth PRM compute).
            # It must NOT be used as final_acc: short integer GTs (DAPO is all
            # integers) almost always appear as intermediate values in a long
            # CoT even when the model ultimately boxes a wrong answer, which
            # would inflate final_acc to 1.0 and poison the last segment's
            # process reward. Use the real reward function (boxed extraction +
            # mathd/sympy grading) -- the same one used to score PRM
            # continuations -- so the final-segment endpoint_acc is consistent
            # with the per-segment ones.
            answer_present = _answer_string_present(response_text, ground_truth)
            final_acc = _score_completion(item.get("data_source"), response_text, ground_truth)

            records.append(
                {
                    "prompt": item["prompt"],
                    "ground_truth": ground_truth,
                    "data_source": item.get("data_source"),
                    "response_text": response_text,
                    "segment_bounds": bounds,
                    "endpoint_accs": {},
                    "final_acc": final_acc,
                    "answer_present": answer_present,
                    "skip_pr": False,
                }
            )

            if not bounds or ground_truth is None or not answer_present:
                records[-1]["skip_pr"] = True

        return records, raw_prompts

    def compute_continuation_pr(
        self,
        records: list[dict[str, Any]],
        raw_prompts: list[list[dict[str, Any]]],
        pr_cfg: ProcessRewardConfig,
        cfg: ConsistencyConfig,
    ) -> list[dict[str, Any]]:
        """Evaluate process rewards on the pre-generated records.

        Each continuation model rebuilds the prefix prompts using its own
        tokenizer, so the same character-level segment boundaries are evaluated.
        """
        # Deep copy so each continuation model gets its own independent results
        # without mutating the shared records.
        records = copy.deepcopy(records)
        prefix_prompt_texts = []
        prompt_meta = []  # (item_idx, segment_idx)

        for i, rec in enumerate(records):
            if rec.get("skip_pr"):
                continue
            for seg_idx, (_start, end) in enumerate(rec["segment_bounds"][:-1]):
                prefix = rec["response_text"][:end]
                prefix_prompt_texts.append(self._build_prefix_prompt(raw_prompts[i], prefix))
                prompt_meta.append((i, seg_idx))

        # Generate K suffixes for each prefix prompt.
        if prefix_prompt_texts:
            prefix_prompt_texts = self._truncate_prompts(prefix_prompt_texts, pr_cfg.max_prompt_length)
            prefix_lengths = self._prompt_lengths(prefix_prompt_texts)
            max_tokens_list = [
                max(1, min(pr_cfg.max_tokens, pr_cfg.max_total_length - int(plen)))
                for plen in prefix_lengths
            ]
            suffix_texts = self.generate(prefix_prompt_texts, pr_cfg, max_tokens_list=max_tokens_list, n=pr_cfg.k_eval)
        else:
            suffix_texts = []

        expected = len(prompt_meta) * pr_cfg.k_eval
        if len(suffix_texts) != expected:
            raise RuntimeError(
                f"Process-reward generation count mismatch: expected {expected}, got {len(suffix_texts)}"
            )

        # Score each suffix and store endpoint accuracies.
        cursor = 0
        for item_idx, endpoint_idx in prompt_meta:
            rec = records[item_idx]
            scores = []
            prefix_end = rec["segment_bounds"][endpoint_idx][1]
            for _ in range(pr_cfg.k_eval):
                full_text = rec["response_text"][:prefix_end] + suffix_texts[cursor]
                cursor += 1
                scores.append(_score_completion(rec["data_source"], full_text, rec["ground_truth"]))
            acc = float(sum(scores) / max(1, len(scores)))
            rec["endpoint_accs"][endpoint_idx] = acc

        # Convert endpoint accuracies into process rewards and merge same-sign segments.
        pr_lists = []
        bounds_lists = []
        for rec in records:
            if rec.get("skip_pr"):
                pr_lists.append([])
                bounds_lists.append([])
                rec["merged_bounds"] = []
                rec["process_rewards"] = []
                rec["raw_bounds"] = list(rec.get("segment_bounds", []))
                rec["raw_process_rewards"] = []
                continue

            prev = 0.0
            prs = []
            raw_bounds = list(rec["segment_bounds"])
            last_seg_idx = len(raw_bounds) - 1
            rec["endpoint_accs"][last_seg_idx] = float(rec.get("final_acc", 0.0))

            for seg_idx, _span in enumerate(raw_bounds):
                acc = float(rec["endpoint_accs"].get(seg_idx, prev))
                prs.append(acc - prev)
                prev = acc

            merged_bounds, merged_prs = _merge_same_sign_segments(raw_bounds, prs)
            pr_lists.append(merged_prs)
            bounds_lists.append(merged_bounds)
            rec["merged_bounds"] = merged_bounds
            rec["process_rewards"] = merged_prs
            rec["raw_bounds"] = raw_bounds
            rec["raw_process_rewards"] = prs

        return records

    def __del__(self):
        # Best-effort cleanup to free GPU memory before loading the next model.
        try:
            del self.llm
            torch.cuda.empty_cache()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main smoke-test entry point
# ---------------------------------------------------------------------------

DATA_PATH = os.environ.get("DATA_PATH", "/data/chenyang/OPD/datasets/dapo-math-17k-processed.parquet")
SKYWORK_MODEL = os.environ.get("SKYWORK_MODEL", "/data/chenyang/models/JustRL-1.5B")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "/data/chenyang/models/DeepSeek-R1-Distill-Qwen-1.5B")
MAIN_RESPONSE_MODEL = os.environ.get("MAIN_RESPONSE_MODEL", DEEPSEEK_MODEL)
N_SAMPLES = int(os.environ.get("N_SAMPLES", "100"))
RANDOM_SEED = int(os.environ.get("RANDOM_SEED", "42"))
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/data/chenyang/OPD/dapo_smoke_test_pr_results.parquet")
GPU_MEMORY_UTILIZATION = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.8"))
MODEL_DTYPE = os.environ.get("MODEL_DTYPE", "bfloat16")
MAIN_RESPONSE_MAX_TOKENS = int(os.environ.get("MAIN_RESPONSE_MAX_TOKENS", "7168"))


def load_data(path: str, n_samples: int, seed: int) -> list[dict[str, Any]]:
    df = pd.read_parquet(path)
    required = {"prompt", "ground_truth"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset {path} is missing columns: {missing}; got {list(df.columns)}")
    df = df.sample(n=min(n_samples, len(df)), random_state=seed).reset_index(drop=True)
    items = []
    for _, row in df.iterrows():
        items.append(
            {
                "prompt": str(row["prompt"]),
                "ground_truth": str(row["ground_truth"]),
                "data_source": row.get("data_source", "sky_candidates"),
            }
        )
    return items


def _cleanup() -> None:
    import gc
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def main() -> None:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    pr_cfg = env_process_reward_config()
    cfg = env_consistency_config()

    print(f"Loading dataset: {DATA_PATH}")
    items = load_data(DATA_PATH, N_SAMPLES, RANDOM_SEED)
    print(f"Sampled {len(items)} items")
    print(f"Main trajectory model: {MAIN_RESPONSE_MODEL}")
    print(f"Continuation models: {SKYWORK_MODEL}, {DEEPSEEK_MODEL}")

    # Generate the shared reasoning trajectories once.
    print(f"\n{'=' * 60}")
    print(f"Generating main trajectories with: {MAIN_RESPONSE_MODEL}")
    print(f"{'=' * 60}")
    main_engine = VLLMContinuationEngine(
        model_path=MAIN_RESPONSE_MODEL,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype=MODEL_DTYPE,
    )
    shared_records, raw_prompts = main_engine.generate_main_responses(
        items, pr_cfg, cfg, main_max_tokens=MAIN_RESPONSE_MAX_TOKENS
    )

    # Evaluate each continuation model.  If the main model is one of the
    # continuation models, reuse the already-loaded engine to avoid loading it
    # twice.
    continuation_models = [SKYWORK_MODEL, DEEPSEEK_MODEL]
    model_records: dict[str, list[dict[str, Any]]] = {}

    if MAIN_RESPONSE_MODEL in continuation_models:
        # Reuse the main engine for its own continuation evaluation.
        main_name = Path(MAIN_RESPONSE_MODEL).name
        print(f"\n{'=' * 60}")
        print(f"Continuation evaluation (main model): {main_name}")
        print(f"{'=' * 60}")
        model_records[main_name] = main_engine.compute_continuation_pr(shared_records, raw_prompts, pr_cfg, cfg)
        del main_engine
        _cleanup()

    for model_path in continuation_models:
        name = Path(model_path).name
        if name in model_records:
            continue
        print(f"\n{'=' * 60}")
        print(f"Continuation evaluation: {name}")
        print(f"{'=' * 60}")
        engine = VLLMContinuationEngine(
            model_path=model_path,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            dtype=MODEL_DTYPE,
        )
        model_records[name] = engine.compute_continuation_pr(shared_records, raw_prompts, pr_cfg, cfg)
        del engine
        _cleanup()

    # Merge results into a single flat dataframe.
    rows = []
    sky_name = Path(SKYWORK_MODEL).name
    ds_name = Path(DEEPSEEK_MODEL).name
    sky_records = model_records[sky_name]
    ds_records = model_records[ds_name]

    for idx, (shared, sky, ds) in enumerate(zip(shared_records, sky_records, ds_records)):
        row = {
            "idx": idx,
            "prompt": shared["prompt"],
            "ground_truth": shared["ground_truth"],
            "main_response_model": Path(MAIN_RESPONSE_MODEL).name,
            "response_text": shared["response_text"],
            "answer_present_in_response": shared["answer_present"],
            f"merged_bounds_{sky_name}": json.dumps(sky.get("merged_bounds", [])),
            f"merged_bounds_{ds_name}": json.dumps(ds.get("merged_bounds", [])),
            f"process_rewards_{sky_name}": json.dumps(sky.get("process_rewards", [])),
            f"process_rewards_{ds_name}": json.dumps(ds.get("process_rewards", [])),
            f"raw_bounds_{sky_name}": json.dumps(sky.get("raw_bounds", [])),
            f"raw_bounds_{ds_name}": json.dumps(ds.get("raw_bounds", [])),
            f"raw_process_rewards_{sky_name}": json.dumps(sky.get("raw_process_rewards", [])),
            f"raw_process_rewards_{ds_name}": json.dumps(ds.get("raw_process_rewards", [])),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved results to: {OUTPUT_PATH}")
    print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
    print("\nPreview:")
    print(df.head().to_string())


if __name__ == "__main__":
    main()
