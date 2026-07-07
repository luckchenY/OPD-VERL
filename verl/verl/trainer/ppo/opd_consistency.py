"""Batch-level OPD process-reward/KL consistency pipeline.

The new verl training path already has the online order

    rollout -> top-k student/teacher KL reward -> PPO update

This module inserts the missing OPD reliability-filtering steps between KL reward
construction and PPO advantage computation:

    rollout responses
    -> segment responses
    -> compute process rewards from K-fold prefix continuations
    -> aggregate existing top-k KL reward per segment
    -> compute PR/KL ordering consistency
    -> mask/drop unreliable KL reward

The trainer integration is intentionally narrow: ``ray_trainer.py`` calls only
``run_opd_consistency_pipeline`` after ``compute_distillation_reward`` has
populated ``rm_scores``.
"""

from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Iterable

import numpy as np
import torch


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


@dataclass
class ProcessRewardConfig:
    enable: bool = False
    k_eval: int = 4
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 1.0
    max_tokens: int = 384
    max_prompt_length: int = 8192
    max_total_length: int = 8192
    batch_size: int = 0
    stop: tuple[str, ...] = ("<|im_end|>",)


@dataclass
class ConsistencyConfig:
    enable: bool = False
    top_percent_responses: float = 30.0
    mask_top_percent_segments: float = 20.0
    min_segments_per_response: int = 3
    min_sentences_per_segment: int = 5
    drop_low_consistency: bool = True
    strict_pr_required: bool = True
    max_segments: int = 128
    process_reward: ProcessRewardConfig = field(default_factory=ProcessRewardConfig)


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


def _cfg_get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _read_process_reward_config(raw: Any) -> ProcessRewardConfig:
    return ProcessRewardConfig(
        enable=_as_bool(_cfg_get(raw, "enable", False)),
        k_eval=max(1, int(_cfg_get(raw, "k_eval", 4))),
        temperature=float(_cfg_get(raw, "temperature", 0.7)),
        top_k=int(_cfg_get(raw, "top_k", 50)),
        top_p=float(_cfg_get(raw, "top_p", 1.0)),
        max_tokens=int(_cfg_get(raw, "max_tokens", 384)),
        max_prompt_length=int(_cfg_get(raw, "max_prompt_length", 8192)),
        max_total_length=int(_cfg_get(raw, "max_total_length", _cfg_get(raw, "max_prompt_length", 8192))),
        batch_size=int(_cfg_get(raw, "batch_size", 0)),
        stop=tuple(_cfg_get(raw, "stop", ("<|im_end|>",)) or ()),
    )


def _env_opd_config() -> ConsistencyConfig:
    pr_raw = {
        "enable": os.environ.get("OPD_PROCESS_REWARD_ENABLE", os.environ.get("OPD_CONSISTENCY_ENABLE", "False")),
        "k_eval": os.environ.get("OPD_PROCESS_REWARD_K_EVAL", "4"),
        "temperature": os.environ.get("OPD_PROCESS_REWARD_TEMPERATURE", "0.7"),
        "top_k": os.environ.get("OPD_PROCESS_REWARD_TOP_K", "50"),
        "top_p": os.environ.get("OPD_PROCESS_REWARD_TOP_P", "1.0"),
        "max_tokens": os.environ.get("OPD_PROCESS_REWARD_MAX_TOKENS", "384"),
        "max_prompt_length": os.environ.get("OPD_PROCESS_REWARD_MAX_PROMPT_LENGTH", "8192"),
        "max_total_length": os.environ.get("OPD_PROCESS_REWARD_MAX_TOTAL_LENGTH", "8192"),
        "batch_size": os.environ.get("OPD_PROCESS_REWARD_BATCH_SIZE", "0"),
    }
    return ConsistencyConfig(
        enable=_as_bool(os.environ.get("OPD_CONSISTENCY_ENABLE", "False")),
        top_percent_responses=float(os.environ.get("OPD_CONSISTENCY_TOP_PERCENT_RESPONSES", "30.0")),
        mask_top_percent_segments=float(os.environ.get("OPD_CONSISTENCY_MASK_TOP_PERCENT_SEGMENTS", "20.0")),
        min_segments_per_response=int(os.environ.get("OPD_CONSISTENCY_MIN_SEGMENTS", "3")),
        min_sentences_per_segment=int(os.environ.get("OPD_SEGMENT_MIN_SENTENCES", "5")),
        drop_low_consistency=_as_bool(os.environ.get("OPD_CONSISTENCY_DROP_LOW", "True"), True),
        strict_pr_required=_as_bool(os.environ.get("OPD_CONSISTENCY_STRICT_PR_REQUIRED", "True"), True),
        max_segments=int(os.environ.get("OPD_CONSISTENCY_MAX_SEGMENTS", "128")),
        process_reward=_read_process_reward_config(pr_raw),
    )


def load_consistency_config(config: Any) -> ConsistencyConfig:
    """Read OPD options from a safe namespace or OPD_* environment variables."""
    algorithm_cfg = _cfg_get(config, "algorithm")
    raw = _cfg_get(algorithm_cfg, "opd_consistency", None)

    if raw is None:
        rollout_cfg = _cfg_get(_cfg_get(config, "actor_rollout_ref"), "rollout")
        raw = _cfg_get(rollout_cfg, "opd_consistency", None)

    if raw is None:
        return _env_opd_config()

    return ConsistencyConfig(
        enable=_as_bool(_cfg_get(raw, "enable", False)),
        top_percent_responses=float(_cfg_get(raw, "top_percent_responses", 30.0)),
        mask_top_percent_segments=float(_cfg_get(raw, "mask_top_percent_segments", 20.0)),
        min_segments_per_response=int(_cfg_get(raw, "min_segments_per_response", 3)),
        min_sentences_per_segment=int(_cfg_get(raw, "min_sentences_per_segment", 5)),
        drop_low_consistency=_as_bool(_cfg_get(raw, "drop_low_consistency", True), True),
        strict_pr_required=_as_bool(_cfg_get(raw, "strict_pr_required", True), True),
        max_segments=int(_cfg_get(raw, "max_segments", 128)),
        process_reward=_read_process_reward_config(_cfg_get(raw, "process_reward", {})),
    )

def _to_python_item(value: Any, idx: int | None = None) -> Any:
    if idx is not None:
        try:
            value = value[idx]
        except Exception:
            return None
    if isinstance(value, np.ndarray) and value.shape == ():
        return value.item()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    return value


def _lookup_nested(mapping: Any, keys: Iterable[str]) -> Any:
    cur = mapping
    for key in keys:
        if cur is None:
            return None
        if isinstance(cur, dict):
            cur = cur.get(key)
        else:
            cur = getattr(cur, key, None)
    return cur


def _coerce_float_list(value: Any) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for item in value:
        try:
            out.append(float(item))
        except Exception:
            return None
    return out


def _coerce_bounds(value: Any) -> list[tuple[int, int]] | None:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, (list, tuple)):
        return None
    out = []
    for item in value:
        if isinstance(item, dict):
            start = item.get("start_char", item.get("start", None))
            end = item.get("end_char", item.get("end", None))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            start, end = item[0], item[1]
        else:
            return None
        try:
            start_i, end_i = int(start), int(end)
        except Exception:
            return None
        if end_i > start_i:
            out.append((start_i, end_i))
    return out or None


def _sentence_count(text: str) -> int:
    return text.count(".")


def _segment_text(text: str, max_segments: int, min_sentences_per_segment: int = 5) -> list[tuple[int, int]]:
    """Split reasoning at old Swift trigger words, avoiding short segments."""
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

def _decode_response_and_token_spans(tokenizer: Any, token_ids: torch.Tensor, valid_len: int) -> tuple[str, list[tuple[int, int]]]:
    ids = token_ids[:valid_len].detach().cpu().tolist()
    pieces: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for tok_id in ids:
        piece = tokenizer.decode([int(tok_id)], skip_special_tokens=True)
        start = cursor
        cursor += len(piece)
        pieces.append(piece)
        spans.append((start, cursor))
    return "".join(pieces), spans


def _char_span_to_token_slice(char_span: tuple[int, int], token_spans: list[tuple[int, int]]) -> tuple[int, int] | None:
    start_char, end_char = char_span
    token_indices = [i for i, (s, e) in enumerate(token_spans) if e > start_char and s < end_char]
    if not token_indices:
        return None
    return token_indices[0], token_indices[-1] + 1


def _extract_process_rewards(batch: Any, item_idx: int) -> list[float] | None:
    names = ("segment_process_rewards", "process_rewards", "pr_scores", "opd_process_rewards", "opd_pr_scores")
    nt = getattr(batch, "non_tensor_batch", {})
    for name in names:
        if name in nt:
            value = _to_python_item(nt[name], item_idx)
            if value is not None:
                return _coerce_float_list(value)
    extra = _to_python_item(nt.get("extra_info"), item_idx) if "extra_info" in nt else None
    if isinstance(extra, dict):
        for name in names:
            value = extra.get(name)
            if value is not None:
                return _coerce_float_list(value)
        value = _lookup_nested(extra, ("opd", "process_rewards"))
        if value is not None:
            return _coerce_float_list(value)
    return None


def _extract_segment_bounds(batch: Any, item_idx: int) -> list[tuple[int, int]] | None:
    names = ("segment_char_bounds", "segment_bounds", "opd_segment_bounds")
    nt = getattr(batch, "non_tensor_batch", {})
    for name in names:
        if name in nt:
            coerced = _coerce_bounds(_to_python_item(nt[name], item_idx))
            if coerced:
                return coerced
    extra = _to_python_item(nt.get("extra_info"), item_idx) if "extra_info" in nt else None
    if isinstance(extra, dict):
        for name in names:
            coerced = _coerce_bounds(extra.get(name))
            if coerced:
                return coerced
        coerced = _coerce_bounds(_lookup_nested(extra, ("opd", "segment_bounds")))
        if coerced:
            return coerced
    return None


def _extract_data_source_and_gt(batch: Any, item_idx: int) -> tuple[Any, Any]:
    nt = getattr(batch, "non_tensor_batch", {})
    data_source = _to_python_item(nt.get("data_source"), item_idx) if "data_source" in nt else None
    reward_model = _to_python_item(nt.get("reward_model"), item_idx) if "reward_model" in nt else None
    ground_truth = None
    if isinstance(reward_model, dict):
        ground_truth = reward_model.get("ground_truth")
    return data_source, ground_truth


def _get_raw_prompt(batch: Any, item_idx: int) -> list[dict[str, Any]] | None:
    nt = getattr(batch, "non_tensor_batch", {})
    raw = _to_python_item(nt.get("raw_prompt"), item_idx) if "raw_prompt" in nt else None
    if raw is None and "prompt" in nt:
        raw = _to_python_item(nt.get("prompt"), item_idx)
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, list):
        return list(raw)
    return None


def _build_prefix_prompt(tokenizer: Any, raw_prompt: list[dict[str, Any]] | None, response_prefix: str) -> str:
    if raw_prompt:
        prompt = tokenizer.apply_chat_template(raw_prompt, add_generation_prompt=True, tokenize=False)
    else:
        prompt = ""
    return prompt + (response_prefix or "") + EPISODE_EVAL_SUFFIX


def _load_raw_reward_fn(config: Any):
    reward_cfg = _cfg_get(config, "custom_reward_function", {})
    file_path = _cfg_get(reward_cfg, "path", None)
    function_name = _cfg_get(reward_cfg, "name", None)
    if not file_path or not function_name:
        return None
    if not os.path.isabs(file_path):
        file_path = os.path.abspath(file_path)
    spec = importlib.util.spec_from_file_location("opd_custom_reward_module", file_path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    raw_fn = getattr(module, function_name, None)
    if raw_fn is None:
        return None
    reward_kwargs = dict(_cfg_get(reward_cfg, "reward_kwargs", {}) or {})
    return partial(_call_reward_with_kwargs, raw_fn, reward_kwargs)


def _call_reward_with_kwargs(raw_fn, reward_kwargs, *args, **kwargs):
    merged = {**kwargs, **reward_kwargs}
    return raw_fn(*args, **merged)


def _score_completion(raw_reward_fn, data_source: Any, text: str, ground_truth: Any) -> float:
    if raw_reward_fn is None or ground_truth is None:
        return 0.0
    score = raw_reward_fn(data_source=data_source, solution_str=text, ground_truth=ground_truth, extra_info={})
    if isinstance(score, dict):
        return float(score.get("score", 0.0))
    if isinstance(score, (tuple, list)):
        return float(score[0]) if score else 0.0
    return float(score)


def _answer_string_present(text: str, ground_truth: Any) -> bool:
    """Cheap math-answer precheck before expensive K-fold PR evaluation."""
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


def _merge_same_sign_segments(
    bounds: list[tuple[int, int]],
    process_rewards: list[float],
) -> tuple[list[tuple[int, int]], list[float]]:
    """Merge adjacent segments while the PR sign does not flip.

    Zero-PR segments are treated as not changing the current sign, so they are
    absorbed into the adjacent run where possible.
    """
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


def _build_generation_dataproto(tokenizer: Any, prompt_texts: list[str], max_prompt_length: int):
    from verl import DataProto

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    position_ids = attention_mask.cumsum(dim=-1) - 1
    position_ids = torch.clamp(position_ids, min=0)
    raw_prompt_ids = []
    for ids, mask in zip(input_ids, attention_mask):
        raw_prompt_ids.append(ids[mask.bool()].tolist())
    return DataProto.from_dict(
        tensors={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        },
        non_tensors={"raw_prompt_ids": np.array(raw_prompt_ids, dtype=object)},
        meta_info={"eos_token_id": tokenizer.eos_token_id},
    )


def _decode_generation_output(tokenizer: Any, output: Any) -> list[str]:
    responses = output.batch["responses"].detach().cpu()
    masks = output.batch.get("response_mask")
    masks = masks.detach().cpu() if masks is not None else torch.ones_like(responses)
    texts = []
    for ids, mask in zip(responses, masks):
        valid_len = int(mask.sum().item())
        texts.append(tokenizer.decode(ids[:valid_len].tolist(), skip_special_tokens=True))
    return texts


def _run_pr_generations(actor_rollout_wg: Any, tokenizer: Any, prompt_texts: list[str], pr_cfg: ProcessRewardConfig):
    if not prompt_texts:
        return []

    k_eval = max(1, int(pr_cfg.k_eval))
    max_total_length = max(1, int(pr_cfg.max_total_length))
    max_prompt_length = min(max(1, int(pr_cfg.max_prompt_length)), max_total_length)
    world_size = max(1, int(getattr(actor_rollout_wg, "world_size", 1) or 1))
    configured_prefix_batch_size = int(pr_cfg.batch_size or 0)

    buckets: list[list[str]] = [[] for _ in prompt_texts]
    valid_items: list[tuple[int, str, int]] = []

    enc = tokenizer(
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    prompt_lens = enc["attention_mask"].sum(dim=-1).tolist()
    for prompt_idx, (prompt_text, prompt_len) in enumerate(zip(prompt_texts, prompt_lens, strict=True)):
        available_tokens = max_total_length - int(prompt_len)
        if available_tokens <= 0:
            buckets[prompt_idx] = [""] * k_eval
            continue
        max_tokens = max(1, min(int(pr_cfg.max_tokens), available_tokens))
        valid_items.append((prompt_idx, prompt_text, max_tokens))

    if configured_prefix_batch_size > 0:
        prefix_batch_size = max(1, configured_prefix_batch_size)
    else:
        prefix_batch_size = max(1, len(valid_items))

    print(
        f"[opd_pr] generation_start prompts={len(prompt_texts)} k_eval={k_eval} "
        f"world_size={world_size} prefix_batch_size={prefix_batch_size} "
        f"max_total_length={max_total_length} max_prompt_length={max_prompt_length} max_tokens={pr_cfg.max_tokens}",
        flush=True,
    )
    for start_idx in range(0, len(valid_items), prefix_batch_size):
        group = valid_items[start_idx : start_idx + prefix_batch_size]
        if not group:
            continue
        group_prompts: list[str] = []
        group_owners: list[int | None] = []
        group_max_token_list: list[int] = []
        for prompt_idx, prompt_text, max_tokens in group:
            group_prompts.append(prompt_text)
            group_owners.append(prompt_idx)
            group_max_token_list.append(int(max_tokens))

        padded_count = int(math.ceil(len(group_prompts) / world_size) * world_size)
        if padded_count > len(group_prompts):
            pad_prompt = group_prompts[-1]
            pad_max_tokens = group_max_token_list[-1]
            group_prompts.extend([pad_prompt] * (padded_count - len(group_prompts)))
            group_owners.extend([None] * (padded_count - len(group_owners)))
            group_max_token_list.extend([pad_max_tokens] * (padded_count - len(group_max_token_list)))
        group_max_tokens = max(group_max_token_list)

        print(
            f"[opd_pr] generation_group start={start_idx} prefixes={len(group)} "
            f"padded={len(group_prompts)} n={k_eval} group_max_tokens={group_max_tokens}",
            flush=True,
        )
        prompt_batch = _build_generation_dataproto(tokenizer, group_prompts, max_prompt_length)
        prompt_batch.non_tensor_batch["opd_per_prompt_max_tokens"] = np.array(group_max_token_list, dtype=object)
        prompt_batch.meta_info["opd_sampling_params"] = {
            "n": k_eval,
            "temperature": pr_cfg.temperature,
            "top_k": pr_cfg.top_k,
            "top_p": pr_cfg.top_p,
            "max_tokens": group_max_tokens,
            "logprobs": None,
            "use_tqdm": True,
        }
        output = actor_rollout_wg.generate_sequences(prompt_batch)
        print(f"[opd_pr] generation_group_done start={start_idx}", flush=True)
        decoded = _decode_generation_output(tokenizer, output)
        decoded_owners: list[int | None] = []
        for owner in group_owners:
            decoded_owners.extend([owner] * k_eval)
        for owner, suffix in zip(decoded_owners, decoded, strict=False):
            if owner is not None and len(buckets[owner]) < k_eval:
                buckets[owner].append(suffix)

    texts: list[str] = []
    for bucket in buckets:
        if len(bucket) < k_eval:
            bucket = bucket + [""] * (k_eval - len(bucket))
        texts.extend(bucket[:k_eval])
    return texts

def compute_process_rewards_for_batch(batch: Any, tokenizer: Any, config: Any, actor_rollout_wg: Any, global_step: int | None = None) -> tuple[Any, dict[str, float]]:
    """Segment responses and compute old-style K-fold endpoint PR in-batch."""
    cfg = load_consistency_config(config)
    pr_cfg = cfg.process_reward
    if not (cfg.enable and pr_cfg.enable):
        return batch, {}
    if actor_rollout_wg is None:
        return batch, {"opd_pr/skipped_no_rollout_worker": 1.0}

    raw_reward_fn = _load_raw_reward_fn(config)
    if raw_reward_fn is None:
        return batch, {"opd_pr/skipped_no_reward_fn": 1.0}

    responses = batch.batch["responses"]
    response_mask = batch.batch["response_mask"]

    records = []
    prompt_texts = []
    prompt_meta = []
    n_answer_absent = 0
    n_answer_present = 0
    n_final_correct = 0

    for i in range(responses.shape[0]):
        valid_len = int(response_mask[i].detach().sum().item())
        response_text, _ = _decode_response_and_token_spans(tokenizer, responses[i], valid_len)
        bounds = _extract_segment_bounds(batch, i) or _segment_text(response_text, cfg.max_segments, cfg.min_sentences_per_segment)
        bounds = bounds[: cfg.max_segments]
        raw_prompt = _get_raw_prompt(batch, i)
        data_source, gt = _extract_data_source_and_gt(batch, i)
        answer_present = _answer_string_present(response_text, gt)
        final_acc = 1.0 if answer_present else 0.0
        rec = {
            "response_text": response_text,
            "segment_bounds": bounds,
            "data_source": data_source,
            "ground_truth": gt,
            "endpoint_accs": {},
            "final_acc": final_acc,
            "answer_present": answer_present,
            "skip_pr": False,
        }
        records.append(rec)

        if not bounds or gt is None or not answer_present:
            rec["skip_pr"] = True
            n_answer_absent += int(not answer_present)
            continue

        n_answer_present += 1
        n_final_correct += int(final_acc > 0.5)
        last_seg_idx = len(bounds) - 1
        rec["endpoint_accs"][last_seg_idx] = final_acc

        # Only validate continuations up to the penultimate segment. The final
        # segment's endpoint accuracy is the final answer correctness itself.
        for seg_idx, (_start, end) in enumerate(bounds[:-1]):
            prefix = response_text[:end]
            prompt_texts.append(_build_prefix_prompt(tokenizer, raw_prompt, prefix))
            prompt_meta.append((i, seg_idx, prefix))

    print(f"[opd_pr] prompt_meta={len(prompt_meta)} records={len(records)} k_eval={pr_cfg.k_eval}", flush=True)
    suffix_texts = _run_pr_generations(actor_rollout_wg, tokenizer, prompt_texts, pr_cfg)
    expected = len(prompt_meta) * pr_cfg.k_eval
    if len(suffix_texts) != expected:
        return batch, {
            "opd_pr/skipped_bad_generation_count": 1.0,
            "opd_pr/prompts": float(len(prompt_meta)),
            "opd_pr/generated": float(len(suffix_texts)),
        }

    cursor = 0
    for item_idx, endpoint_idx, prefix in prompt_meta:
        rec = records[item_idx]
        scores = []
        for _ in range(pr_cfg.k_eval):
            full_text = prefix + suffix_texts[cursor]
            cursor += 1
            scores.append(_score_completion(raw_reward_fn, rec["data_source"], full_text, rec["ground_truth"]))
        acc = float(sum(scores) / max(1, len(scores)))
        rec["endpoint_accs"][endpoint_idx] = acc

    pr_lists = []
    bounds_lists = []
    n_raw_segments = 0
    n_merged_segments = 0
    for rec in records:
        if rec.get("skip_pr"):
            pr_lists.append([])
            bounds_lists.append([])
            continue

        # The first segment's prior correctness is fixed to 0.0.
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
        n_raw_segments += len(prs)
        n_merged_segments += len(merged_prs)

    batch.non_tensor_batch["segment_process_rewards"] = np.array(pr_lists, dtype=object)
    batch.non_tensor_batch["segment_char_bounds"] = np.array(bounds_lists, dtype=object)
    _debug_dump_records(
        "pr",
        [
            {
                "item_idx": idx,
                "answer_present": rec.get("answer_present"),
                "skip_pr": rec.get("skip_pr"),
                "ground_truth": rec.get("ground_truth"),
                "segment_bounds": bounds_lists[idx],
                "process_rewards": pr_lists[idx],
                "endpoint_accs": rec.get("endpoint_accs", {}),
                "response_preview": rec.get("response_text", "")[:2000],
            }
            for idx, rec in enumerate(records)
        ],
        global_step=global_step,
    )
    return batch, {
        "opd_pr/enabled": 1.0,
        "opd_pr/prompts": float(len(prompt_meta)),
        "opd_pr/k_eval": float(pr_cfg.k_eval),
        "opd_pr/answer_present": float(n_answer_present),
        "opd_pr/answer_absent_fallback": float(n_answer_absent),
        "opd_pr/final_correct_direct": float(n_final_correct),
        "opd_pr/raw_segments": float(n_raw_segments),
        "opd_pr/merged_segments": float(n_merged_segments),
    }


def _violation(pr_a: float, kl_a: float, pr_b: float, kl_b: float) -> float:
    pr_gap = pr_a - pr_b
    kl_inv = kl_a - kl_b
    if pr_gap <= 0.0 or kl_inv <= 0.0:
        return 0.0
    return float(pr_gap * kl_inv)


def _score_segments(process_rewards: list[float], norm_kls: list[float]) -> tuple[list[float], float, int]:
    n = min(len(process_rewards), len(norm_kls))
    inconsistency = [0.0] * n
    if n < 3:
        # Segment 0 is conditioning context and never participates in consistency.
        # Need at least two later segments to form one comparison pair.
        return inconsistency, math.nan, 0

    # Exclude the first segment from violation scoring entirely. This also
    # prevents the second segment from ever being compared against the first.
    comparable = [idx for idx in range(1, n)]
    order = sorted(comparable, key=lambda i: (-process_rewards[i], norm_kls[i]))
    n_pairs = 0
    n_consistent = 0
    for rank in range(len(order) - 1):
        i = order[rank]
        j = order[rank + 1]
        if i == 0 or j == 0:
            continue
        pr_a, pr_b = process_rewards[i], process_rewards[j]
        if pr_a <= pr_b:
            continue
        kl_a, kl_b = norm_kls[i], norm_kls[j]
        n_pairs += 1
        if kl_a <= kl_b:
            n_consistent += 1
        v = _violation(pr_a, kl_a, pr_b, kl_b)
        inconsistency[i] += v
        inconsistency[j] += v
    if n_pairs == 0:
        return inconsistency, math.nan, 0
    return inconsistency, n_consistent / n_pairs, n_pairs

def _token_kl_from_rm_scores(rm_scores: torch.Tensor) -> torch.Tensor:
    if rm_scores.ndim == 3:
        return -rm_scores.detach().sum(dim=-1)
    return -rm_scores.detach()


def _debug_dump_records(kind: str, records: list[dict[str, Any]], global_step: int | None = None) -> None:
    debug_dir = os.environ.get("OPD_DEBUG_DIR")
    if not debug_dir:
        return
    try:
        os.makedirs(debug_dir, exist_ok=True)
        max_records = int(os.environ.get("OPD_DEBUG_MAX_RECORDS", "64"))
        step = "none" if global_step is None else str(global_step)
        path = os.path.join(debug_dir, f"{kind}_step_{step}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            for rec in records[:max_records]:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        print(f"[OPD debug] failed to dump {kind}: {exc}")


def apply_opd_consistency_mask(batch: Any, tokenizer: Any, config: Any, global_step: int | None = None):
    cfg = load_consistency_config(config)
    if not cfg.enable:
        return batch, {}
    if "rm_scores" not in batch.batch or "responses" not in batch.batch:
        return batch, {"opd_consistency/enabled": 1.0, "opd_consistency/skipped_no_rm_scores": 1.0}

    rm_scores = batch.batch["rm_scores"]
    responses = batch.batch["responses"]
    response_mask = batch.batch.get("response_mask")
    if response_mask is None:
        return batch, {"opd_consistency/enabled": 1.0, "opd_consistency/skipped_no_response_mask": 1.0}

    token_kl = _token_kl_from_rm_scores(rm_scores).to(response_mask.device)
    bsz, response_len = responses.shape[0], responses.shape[1]
    keep_masks = torch.ones((bsz, response_len), dtype=rm_scores.dtype, device=rm_scores.device)

    annotated = []
    n_missing_pr = 0
    n_segments_total = 0
    n_pairs_total = 0

    for i in range(bsz):
        valid_len = int(response_mask[i].detach().sum().item())
        if valid_len <= 0:
            annotated.append(None)
            continue
        pr_values = _extract_process_rewards(batch, i)
        if not pr_values:
            n_missing_pr += 1
            annotated.append(None)
            continue
        text, token_spans = _decode_response_and_token_spans(tokenizer, responses[i], valid_len)
        char_bounds = _extract_segment_bounds(batch, i) or _segment_text(text, cfg.max_segments, cfg.min_sentences_per_segment)
        n = min(len(char_bounds), len(pr_values), cfg.max_segments)
        if n <= 0:
            annotated.append(None)
            continue
        char_bounds = char_bounds[:n]
        pr_values = pr_values[:n]

        token_slices = []
        norm_kls = []
        prev_token_end = 0
        for span in char_bounds:
            token_slice = _char_span_to_token_slice(span, token_spans)
            if token_slice is None:
                token_slices.append(None)
                norm_kls.append(float("nan"))
                continue
            start, end = token_slice
            start = max(start, prev_token_end)
            if end <= start:
                token_slices.append(None)
                norm_kls.append(float("nan"))
                continue
            prev_token_end = end
            seg_mask = response_mask[i, start:end].bool()
            val = token_kl[i, start:end][seg_mask].float().mean().item() if seg_mask.any() else float("nan")
            token_slices.append((start, end))
            norm_kls.append(val)

        valid = [idx for idx, val in enumerate(norm_kls) if not math.isnan(val)]
        if len(valid) != len(norm_kls):
            pr_values = [pr_values[idx] for idx in valid]
            norm_kls = [norm_kls[idx] for idx in valid]
            token_slices = [token_slices[idx] for idx in valid]
        if len(pr_values) < max(2, cfg.min_segments_per_response):
            annotated.append(None)
            continue
        inconsistency, consistency, n_pairs = _score_segments(pr_values, norm_kls)
        if math.isnan(consistency) or n_pairs == 0:
            annotated.append(None)
            continue
        n_segments_total += len(pr_values)
        n_pairs_total += n_pairs
        annotated.append({
            "idx": i,
            "consistency": consistency,
            "n_pairs": n_pairs,
            "process_rewards": pr_values,
            "norm_kls": norm_kls,
            "inconsistency": inconsistency,
            "token_slices": token_slices,
        })

    eligible = [a for a in annotated if a is not None]
    if not eligible:
        metrics = {
            "opd_consistency/enabled": 1.0,
            "opd_consistency/eligible_responses": 0.0,
            "opd_consistency/missing_pr_responses": float(n_missing_pr),
        }
        if cfg.strict_pr_required:
            metrics["opd_consistency/noop_missing_pr"] = 1.0
        return batch, metrics

    n_keep = max(1, int(math.ceil(len(eligible) * cfg.top_percent_responses / 100.0)))
    ranked = sorted(eligible, key=lambda x: (x["consistency"], x["n_pairs"]), reverse=True)
    kept_indices = {a["idx"] for a in ranked[:n_keep]}

    n_masked_segments = 0
    n_masked_tokens = 0
    n_dropped_responses = 0
    kept_consistency = []
    all_consistency = []
    masked_slices_by_idx: dict[int, list[tuple[int, int]]] = {}

    for a in eligible:
        idx = a["idx"]
        all_consistency.append(a["consistency"])
        if idx not in kept_indices:
            if cfg.drop_low_consistency:
                valid_len = int(response_mask[idx].detach().sum().item())
                keep_masks[idx, :valid_len] = 0.0
                n_dropped_responses += 1
            continue
        kept_consistency.append(a["consistency"])
        n_seg = len(a["token_slices"])
        n_mask_target = min(n_seg, int(math.ceil(n_seg * cfg.mask_top_percent_segments / 100.0)))
        # Segment 0 is a trusted conditioning prefix: never mask it.
        mask_candidates = list(range(1, n_seg))
        ranked_segments = sorted(mask_candidates, key=lambda j: a["inconsistency"][j], reverse=True)
        for seg_idx in ranked_segments[:n_mask_target]:
            if a["inconsistency"][seg_idx] <= 0.0:
                continue
            token_slice = a["token_slices"][seg_idx]
            if token_slice is None:
                continue
            start, end = token_slice
            keep_masks[idx, start:end] = 0.0
            masked_slices_by_idx.setdefault(idx, []).append((start, end))
            n_masked_segments += 1
            n_masked_tokens += max(0, end - start)

    batch.batch["rm_scores"] = rm_scores * (keep_masks.unsqueeze(-1) if rm_scores.ndim == 3 else keep_masks)
    batch.batch["opd_consistency_mask"] = keep_masks
    metrics = {
        "opd_consistency/enabled": 1.0,
        "opd_consistency/eligible_responses": float(len(eligible)),
        "opd_consistency/kept_responses": float(len(kept_indices)),
        "opd_consistency/dropped_responses": float(n_dropped_responses),
        "opd_consistency/missing_pr_responses": float(n_missing_pr),
        "opd_consistency/segments": float(n_segments_total),
        "opd_consistency/pairs": float(n_pairs_total),
        "opd_consistency/masked_segments": float(n_masked_segments),
        "opd_consistency/masked_tokens": float(n_masked_tokens),
        "opd_consistency/mean_consistency": float(np.mean(all_consistency)) if all_consistency else 0.0,
        "opd_consistency/mean_kept_consistency": float(np.mean(kept_consistency)) if kept_consistency else 0.0,
    }
    if global_step is not None:
        metrics["opd_consistency/global_step"] = float(global_step)
    _debug_dump_records(
        "consistency",
        [
            {
                "item_idx": a["idx"],
                "consistency": a["consistency"],
                "n_pairs": a["n_pairs"],
                "kept_response": a["idx"] in kept_indices,
                "process_rewards": a.get("process_rewards"),
                "norm_kls": a.get("norm_kls"),
                "inconsistency": a.get("inconsistency"),
                "token_slices": a.get("token_slices"),
                "masked_slices": masked_slices_by_idx.get(a["idx"], []),
            }
            for a in eligible
        ],
        global_step=global_step,
    )
    return batch, metrics

def run_opd_consistency_pipeline(
    batch: Any,
    tokenizer: Any,
    config: Any,
    actor_rollout_wg: Any = None,
    global_step: int | None = None,
):
    """Main OPD control flow for one online training batch.

    Expected call site:
      1. rollout has produced ``responses``;
      2. top-k KL distillation has produced ``rm_scores``;
      3. this function computes PR if requested, then consistency masks KL;
      4. reward/advantage/training consume the masked ``rm_scores``.
    """
    cfg = load_consistency_config(config)
    if not cfg.enable:
        return batch, {}

    metrics: dict[str, float] = {"opd_pipeline/enabled": 1.0}
    batch, pr_metrics = compute_process_rewards_for_batch(batch, tokenizer, config, actor_rollout_wg, global_step=global_step)
    metrics.update(pr_metrics)
    batch, consistency_metrics = apply_opd_consistency_mask(batch, tokenizer, config, global_step=global_step)
    metrics.update(consistency_metrics)
    return batch, metrics
