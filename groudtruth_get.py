import argparse
import json
import re
from collections import Counter, defaultdict

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm


BOXED_RE = re.compile(r"\\boxed\s*\{")


def extract_prompt(row_prompt):
    if isinstance(row_prompt, list) and row_prompt:
        return str(row_prompt[0].get("content", "")).strip()
    if hasattr(row_prompt, "tolist"):
        value = row_prompt.tolist()
        if isinstance(value, list) and value:
            return str(value[0].get("content", "")).strip()
    return str(row_prompt).strip()


def last_boxed_content(text):
    if not isinstance(text, str):
        return None

    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None

    start = matches[-1].end()
    depth = 1
    i = start
    while i < len(text):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start:i].strip()
        i += 1
    return None


def normalize_prompt(text):
    return " ".join(str(text).strip().split())


def normalize_answer(ans):
    if ans is None:
        return None
    ans = str(ans).strip()
    ans = ans.replace(",", "")
    ans = " ".join(ans.split())
    return ans


def get_hf_user_and_assistant(row):
    if "messages" in row:
        conv = row["messages"]
        user = next((m.get("content", "") for m in conv if m.get("role") == "user"), "")
        assistant = next((m.get("content", "") for m in conv if m.get("role") == "assistant"), "")
        return str(user).strip(), str(assistant).strip()

    if "conversations" in row:
        conv = row["conversations"]
        user = next((m.get("value", "") for m in conv if m.get("from") in {"human", "user"}), "")
        assistant = next((m.get("value", "") for m in conv if m.get("from") in {"gpt", "assistant"}), "")
        return str(user).strip(), str(assistant).strip()

    raise ValueError(f"Unsupported HF row fields: {list(row.keys())}")


def build_answer_map(dataset_name, target_prompts):
    answer_map = defaultdict(Counter)
    ds = load_dataset(dataset_name, split="train", streaming=True)

    for row in tqdm(ds, desc=f"scan {dataset_name}"):
        user, assistant = get_hf_user_and_assistant(row)
        key = normalize_prompt(user)
        if key not in target_prompts:
            continue

        ans = normalize_answer(last_boxed_content(assistant))
        if ans:
            answer_map[key][ans] += 1

    return answer_map


def choose_answer(counter):
    if not counter:
        return None, "no_boxed_answer"

    ranked = counter.most_common()
    if len(ranked) >= 2 and ranked[0][1] == ranked[1][1] and ranked[0][0] != ranked[1][0]:
        return None, f"ambiguous:{ranked[:5]}"

    return ranked[0][0], "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/OpenThoughts3_opd.parquet")
    parser.add_argument("--output", default="datasets/OpenThoughts3_opd_with_gt.parquet")
    parser.add_argument(
        "--hf-datasets",
        nargs="+",
        default=[
            "lllyx/OpenThought3-Qwen3-4B",
            "open-thoughts/OpenThoughts3-1.2M",
        ],
    )
    parser.add_argument("--report", default="datasets/OpenThoughts3_opd_gt_report.json")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    prompts = [extract_prompt(x) for x in df["prompt"].tolist()]
    keys = [normalize_prompt(x) for x in prompts]
    target_set = set(keys)

    merged_answer_map = defaultdict(Counter)

    for dataset_name in args.hf_datasets:
        partial = build_answer_map(dataset_name, target_set)
        for key, counter in partial.items():
            merged_answer_map[key].update(counter)

    chosen = {}
    errors = {}

    for key in target_set:
        ans, status = choose_answer(merged_answer_map.get(key, Counter()))
        if status == "ok":
            chosen[key] = ans
        else:
            errors[key] = status

    if errors:
        examples = []
        for key, status in list(errors.items())[:20]:
            examples.append({"status": status, "prompt": key[:500]})
        report = {
            "input": args.input,
            "output": args.output,
            "total_rows": len(df),
            "unique_prompts": len(target_set),
            "matched_unique_prompts": len(chosen),
            "failed_unique_prompts": len(errors),
            "examples": examples,
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        raise SystemExit(f"Failed to recover all answers. See report: {args.report}")

    new_reward_model = []
    for key, old_rm in zip(keys, df["reward_model"].tolist()):
        rm = dict(old_rm) if isinstance(old_rm, dict) else {"style": "rule"}
        rm["ground_truth"] = chosen[key]
        rm["style"] = rm.get("style", "rule")
        new_reward_model.append(rm)

    df = df.copy()
    df["reward_model"] = new_reward_model
    df.to_parquet(args.output, index=False)

    report = {
        "input": args.input,
        "output": args.output,
        "total_rows": len(df),
        "unique_prompts": len(target_set),
        "matched_unique_prompts": len(chosen),
        "failed_unique_prompts": 0,
        "hf_datasets": args.hf_datasets,
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()