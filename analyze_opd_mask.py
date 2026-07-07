import re, sys
from collections import defaultdict

path = "/data/chenyang/OPD/logs/run_20260627_021405.log"
pat = re.compile(r"step:(\d+)\s+-\s+(.*?)\n")
field_pat = re.compile(r"([a-zA-Z_][a-zA-Z0-9_/.]*):([-+]?\d*\.?\d+(?:e[-+]?\d+)?)")

steps = {}
with open(path, errors="replace") as f:
    for line in f:
        m = re.search(r"step:(\d+)\s+-\s", line)
        if not m:
            continue
        s = int(m.group(1))
        if s in steps:
            continue
        # extract all key:value pairs in this line
        fields = dict(field_pat.findall(line))
        steps[s] = fields

if not steps:
    print("no step lines found")
    sys.exit(0)

# interesting keys
keys = [
    "opd_pr/answer_present","opd_pr/answer_absent_fallback","opd_pr/final_correct_direct",
    "opd_pr/prompts","opd_pr/raw_segments","opd_pr/merged_segments",
    "opd_consistency/eligible_responses","opd_consistency/kept_responses",
    "opd_consistency/dropped_responses","opd_consistency/missing_pr_responses",
    "opd_consistency/segments","opd_consistency/pairs",
    "opd_consistency/masked_segments","opd_consistency/masked_tokens",
    "opd_consistency/mean_consistency","opd_consistency/mean_kept_consistency",
    "opd_consistency/noop_missing_pr",
]
import statistics as st
def fnum(x):
    try: return float(x)
    except: return None

# Only consider steps that actually ran the pipeline (have opd_pipeline/enabled or opd_pr key)
ran = [s for s,f in steps.items() if "opd_pr/answer_present" in f]
ran.sort()
print(f"total steps with opd metrics: {len(ran)}  (range {ran[0]}..{ran[-1]})")
print()
# Aggregate
sums = defaultdict(float)
for s in ran:
    for k in keys:
        v = fnum(steps[s].get(k))
        if v is not None:
            sums[k] += v
n = len(ran)
print(f"{'metric':45s} {'mean':>10s} {'sum':>12s}")
print("-"*70)
for k in keys:
    vals=[fnum(steps[s].get(k)) for s in ran]
    vals=[v for v in vals if v is not None]
    if not vals:
        print(f"{k:45s} {'NA':>10s}")
        continue
    print(f"{k:45s} {sum(vals)/len(vals):>10.3f} {sum(vals):>12.1f}")

# Distribution of eligible/kept/dropped per step
print()
print("per-step eligible/kept/dropped/missing_pr (first 20 steps):")
print("step  batch  ans_pres  ans_abs  elig  kept  drop  miss_pr  masked_seg  masked_tok  consistency")
for s in ran[:20]:
    f=steps[s]
    batch=16
    print(f"{s:4d}  {batch:5d}  {fnum(f.get('opd_pr/answer_present')) or 0:8.0f}  {fnum(f.get('opd_pr/answer_absent_fallback')) or 0:7.0f}  "
          f"{fnum(f.get('opd_consistency/eligible_responses')) or 0:5.0f}  {fnum(f.get('opd_consistency/kept_responses')) or 0:4.0f}  "
          f"{fnum(f.get('opd_consistency/dropped_responses')) or 0:4.0f}  {fnum(f.get('opd_consistency/missing_pr_responses')) or 0:7.0f}  "
          f"{fnum(f.get('opd_consistency/masked_segments')) or 0:9.0f}  {fnum(f.get('opd_consistency/masked_tokens')) or 0:9.0f}  "
          f"{fnum(f.get('opd_consistency/mean_consistency')) or 0:6.3f}")

# Fraction of steps where eligible==0 (no-op)
no_elig = sum(1 for s in ran if (fnum(steps[s].get('opd_consistency/eligible_responses')) or 0)==0)
print()
print(f"steps with eligible=0 (no mask applied): {no_elig}/{n} = {no_elig/n:.3f}")
# Average batch-level: of 16 responses, how many get missing-PR (untouched), how many dropped, how many kept-with-mask
avg_missing = sum(fnum(steps[s].get('opd_consistency/missing_pr_responses')) or 0 for s in ran)/n
avg_dropped = sum(fnum(steps[s].get('opd_consistency/dropped_responses')) or 0 for s in ran)/n
avg_kept    = sum(fnum(steps[s].get('opd_consistency/kept_responses')) or 0 for s in ran)/n
avg_elig    = sum(fnum(steps[s].get('opd_consistency/eligible_responses')) or 0 for s in ran)/n
print(f"avg per step (batch=16): eligible={avg_elig:.2f}  kept={avg_kept:.2f}  dropped={avg_dropped:.2f}  missing_pr(untouched)={avg_missing:.2f}")
print(f"  -> kept share of batch      = {avg_kept/16:.3f}")
print(f"  -> dropped share of batch   = {avg_dropped/16:.3f}")
print(f"  -> untouched share of batch = {avg_missing/16:.3f}  (no PR -> full KL reward kept)")
