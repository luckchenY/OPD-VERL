import json, glob, os, statistics, re
from collections import defaultdict

ROOT = "/data/chenyang/opdswift/examples/train/test_result"

cases = [
    ("orig",  8192,  f"{ROOT}/DeepSeek-R1-Distill-Qwen-1.5B/20260624_230737/max_new_tokens_8192/native/20260624_230835"),
    ("orig",  16384, f"{ROOT}/DeepSeek-R1-Distill-Qwen-1.5B/20260625_015124/max_new_tokens_16384/native/20260625_015228"),
    ("orig",  32768, f"{ROOT}/DeepSeek-R1-Distill-Qwen-1.5B/20260625_015124/max_new_tokens_32768/native/20260625_015944"),
    ("train", 8192,  f"{ROOT}/merged_hf/opd1250/max_new_tokens_8192/native/20260625_004520"),
    ("train", 16384, f"{ROOT}/merged_hf/opd1250/max_new_tokens_16384/native/20260625_004936"),
    ("train", 32768, f"{ROOT}/merged_hf/opd1250/max_new_tokens_32768/native/20260625_005718"),
]

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

def load(path, ds):
    fp = os.path.join(path, "reviews", "*", f"{ds}_default.jsonl")
    files = glob.glob(fp)
    if not files:
        return []
    recs = []
    with open(files[0]) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ac = r["messages"][1]["content"]
            if isinstance(ac, str):
                ac = [{"type": "text", "content": ac}]
            reasoning_item = next((c for c in ac if isinstance(c, dict) and c.get("type") == "reasoning"), None)
            answer_item = next((c for c in ac if isinstance(c, dict) and c.get("type") == "text"), None)
            reasoning_text = (reasoning_item or {}).get("reasoning", "") or ""
            reasoning_tokens = (reasoning_item or {}).get("reasoning_tokens", 0) or 0
            answer_text = ""
            if answer_item is not None:
                if isinstance(answer_item.get("content"), str):
                    answer_text = answer_item["content"]
                else:
                    answer_text = answer_item.get("content") or answer_item.get("text") or ""
                if not isinstance(answer_text, str):
                    answer_text = json.dumps(answer_text)
            score = r.get("sample_score", {}).get("score", {})
            acc = score.get("value", {}).get("acc", 0)
            extracted = score.get("extracted_prediction", "") or ""
            prediction = score.get("prediction", "") or ""
            # Detect truncation: no boxed answer anywhere AND reasoning extremely long
            full_text = (reasoning_text + "\n" + answer_text)
            boxed_in_answer = len(BOXED_RE.findall(answer_text)) > 0
            boxed_in_pred = "\\boxed{" in prediction
            boxed_in_full = len(BOXED_RE.findall(full_text)) > 0
            recs.append({
                "index": r["index"],
                "group": r.get("sample_score", {}).get("group_id"),
                "acc": acc,
                "rtokens": reasoning_tokens,
                "rlen_chars": len(reasoning_text),
                "alen_chars": len(answer_text),
                "gen_chars": len(reasoning_text) + len(answer_text),
                "extracted": extracted.strip(),
                "empty_extracted": (not extracted.strip()),
                "boxed_in_answer": boxed_in_answer,
                "boxed_in_pred": boxed_in_pred,
                "boxed_in_full": boxed_in_full,
                "answer_text": answer_text,
                "reasoning_text": reasoning_text,
                "prediction": prediction,
            })
    return recs

def summ(recs, cap):
    n = len(recs)
    if n == 0:
        return
    accs = [r["acc"] for r in recs]
    rtok = [r["rtokens"] for r in recs]
    # per-problem pass@4-ish
    by_g = defaultdict(list)
    for r in recs:
        by_g[r["group"]].append(r["acc"])
    passk = sum(1 for v in by_g.values() if max(v) > 0.5) / len(by_g)
    truncated = sum(1 for r in recs if not r["boxed_in_full"])
    no_box_answer = sum(1 for r in recs if not r["boxed_in_answer"])
    no_extract = sum(1 for r in recs if r["empty_extracted"])
    long_ans = sum(1 for r in recs if r["alen_chars"] > 5000)
    rtok_sorted = sorted(rtok)
    def pct(seq, p):
        if not seq: return 0
        k = max(0, min(len(seq)-1, int(round(p/100*(len(seq)-1)))))
        return seq[k]
    print(f"  n={n}  acc={sum(accs)/n:.3f}  pass@4={passk:.3f}")
    print(f"    gen_chars (reasoning+answer): mean={sum(r['gen_chars'] for r in recs)/n:.0f}  p50={pct(sorted(r['gen_chars'] for r in recs),50)}  p90={pct(sorted(r['gen_chars'] for r in recs),90)}  p99={pct(sorted(r['gen_chars'] for r in recs),99)}  max={max(r['gen_chars'] for r in recs)}")
    print(f"    reasoning_chars: mean={sum(r['rlen_chars'] for r in recs)/n:.0f}  p50={pct(sorted(r['rlen_chars'] for r in recs),50)}  p90={pct(sorted(r['rlen_chars'] for r in recs),90)}  max={max(r['rlen_chars'] for r in recs)}")
    print(f"    answer(text)_chars: mean={sum(r['alen_chars'] for r in recs)/n:.0f}  p50={pct(sorted(r['alen_chars'] for r in recs),50)}  p90={pct(sorted(r['alen_chars'] for r in recs),90)}  max={max(r['alen_chars'] for r in recs)}")
    print(f"    NO \\boxed anywhere: {truncated}/{n}={truncated/n:.3f}   | \\boxed in text-part: {n-no_box_answer}/{n}={(n-no_box_answer)/n:.3f}")
    print(f"    empty extracted_prediction: {no_extract}/{n}={no_extract/n:.3f}")
    # near-cap proxy: gen chars in top decile
    gchars_sorted = sorted(r['gen_chars'] for r in recs)
    print(f"    gen_chars p99={pct(gchars_sorted,99)} (cap approx {cap} tokens ~ {cap*4} chars)")
    # Among correct vs incorrect: gen chars
    cor = [r["gen_chars"] for r in recs if r["acc"]>0.5]
    inc = [r["gen_chars"] for r in recs if r["acc"]<=0.5]
    print(f"    gen_chars correct: n={len(cor)} mean={sum(cor)/max(1,len(cor)):.0f}  | wrong: n={len(inc)} mean={sum(inc)/max(1,len(inc)):.0f}")

print("="*80)
for label, cap, path in cases:
    print(f"\n### {label.upper()}  max_new_tokens={cap}  ({os.path.basename(os.path.dirname(path))})")
    for ds in ["aime24","aime25"]:
        recs = load(path, ds)
        print(f"  -- {ds} --")
        summ(recs, cap)
