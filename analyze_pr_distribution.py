import json, glob, os
from collections import Counter

files = sorted(glob.glob("/data/chenyang/OPD/debug/opd_mask_*/pr_step_*.jsonl"))

all_endpoint_accs = []   # per-segment endpoint_acc (non-final segments: K-fold acc; final: answer_present)
final_accs = []
nonfinal_accs = []
prs_all = []             # process rewards (diff of endpoint_accs), per response list
n_records = 0
n_skip = 0
n_answer_present = 0
n_answer_absent = 0
seg_counts = []          # raw segment counts per record (when not skipped)
merged_seg_counts = []   # we can recompute merge
endpoint_acc_seq_examples = []

for f in files:
    for line in open(f):
        line=line.strip()
        if not line: continue
        r=json.loads(line)
        n_records += 1
        if r.get("skip_pr"):
            n_skip += 1
            if r.get("answer_present"):
                n_answer_present += 1
            else:
                n_answer_absent += 1
            continue
        n_answer_present += 1
        ea = r.get("endpoint_accs", {})
        # ea keys are seg_idx (str), values are acc
        # non-final = all keys except max
        if not ea:
            continue
        keys = sorted(int(k) for k in ea.keys())
        last = keys[-1]
        final = float(ea[str(last)])
        final_accs.append(final)
        for k in keys[:-1]:
            nonfinal_accs.append(float(ea[str(k)]))
        # reconstruct PR list = diff along seg order 0..last
        order = sorted(ea.keys(), key=lambda k: int(k))
        vals = [float(ea[k]) for k in order]
        prs = [vals[0]] + [vals[i]-vals[i-1] for i in range(1,len(vals))]
        prs_all.append(prs)
        seg_counts.append(len(order))
        if len(endpoint_acc_seq_examples) < 8:
            endpoint_acc_seq_examples.append({
                "file": os.path.basename(os.path.dirname(f)),
                "item_idx": r.get("item_idx"),
                "ground_truth": r.get("ground_truth"),
                "n_seg": len(order),
                "endpoint_accs_seq": vals,
                "process_rewards": prs,
            })

def dist(vals, label):
    vals = [v for v in vals if v is not None]
    if not vals:
        print(f"{label}: no data"); return
    c = Counter()
    for v in vals:
        if v == 0.0: c["0.0"] += 1
        elif v == 1.0: c["1.0"] += 1
        elif 0 < v < 1: c["(0,1)"] += 1
        else: c["other"] += 1
    n=len(vals)
    print(f"{label}: n={n}  0.0={c['0.0']}({c['0.0']/n:.2%})  1.0={c['1.0']}({c['1.0']/n:.2%})  (0,1)={c['(0,1)']}({c['(0,1)']/n:.2%})  other={c['other']}")
    # finer buckets
    import statistics
    nz = [v for v in vals if v>0]
    print(f"    nonzero mean={sum(nz)/max(1,len(nz)):.3f}  median={statistics.median(nz) if nz else 0}")

print(f"=== records: {n_records}  skip_pr={n_skip}  answer_present={n_answer_present}  answer_absent={n_answer_absent} ===")
print()
dist(final_accs,    "final-segment endpoint_acc (answer_present proxy)")
dist(nonfinal_accs, "non-final endpoint_acc (K-fold continuation acc)")
print()
# process reward distribution (all PR values across all responses)
flat_prs = [p for prs in prs_all for p in prs]
dist(flat_prs, "all process_reward values (acc diff)")
print()
# how many segments per response (raw)
import statistics
print(f"raw segments per PR'd response: n={len(seg_counts)} mean={sum(seg_counts)/len(seg_counts):.2f} median={statistics.median(seg_counts)} min={min(seg_counts)} max={max(seg_counts)}")
# after merge same-sign: recompute
def merge_same_sign(prs):
    if not prs: return []
    merged=[prs[0]]
    for p in prs[1:]:
        cur=merged[-1]
        cur_sign = 1 if cur>0 else (-1 if cur<0 else 0)
        s = 1 if p>0 else (-1 if p<0 else 0)
        if cur_sign==0:
            merged[-1]=p
        elif s==0 or s==cur_sign:
            merged[-1]=cur+p
        else:
            merged.append(p)
    return merged
merged_counts=[len(merge_same_sign(prs)) for prs in prs_all]
print(f"merged segments per PR'd response: n={len(merged_counts)} mean={sum(merged_counts)/len(merged_counts):.2f} median={statistics.median(merged_counts)} min={min(merged_counts)} max={max(merged_counts)}")
# how many pass min_segments=5
print(f"  pass min_segments>=5 (merged): {sum(1 for c in merged_counts if c>=5)}/{len(merged_counts)} = {sum(1 for c in merged_counts if c>=5)/len(merged_counts):.2%}")
print(f"  pass min_segments>=3 (merged): {sum(1 for c in merged_counts if c>=3)}/{len(merged_counts)} = {sum(1 for c in merged_counts if c>=3)/len(merged_counts):.2%}")
print(f"  pass min_segments>=2 (merged): {sum(1 for c in merged_counts if c>=2)}/{len(merged_counts)} = {sum(1 for c in merged_counts if c>=2)/len(merged_counts):.2%}")

# How varied is PR within a single response? count responses with >=2 distinct nonzero PR values
def variety(prs):
    nz=sorted(set(round(p,3) for p in prs if abs(p)>1e-9))
    return len(nz)
var_dist = Counter(variety(prs) for prs in prs_all)
print()
print("distinct-nonzero-PR-values per response distribution:", dict(sorted(var_dist.items())))
# fraction of responses where PR is essentially constant (0 or 1 distinct nonzero)
flat = sum(1 for prs in prs_all if variety(prs)<=1)
print(f"  responses with <=1 distinct nonzero PR: {flat}/{len(prs_all)} = {flat/len(prs_all):.2%}")

print()
print("=== example endpoint_accs / process_rewards sequences ===")
for e in endpoint_acc_seq_examples:
    print(f"  [{e['file']} idx={e['item_idx']} gt={e['ground_truth']} n_seg={e['n_seg']}]")
    print(f"     endpoint_accs: {e['endpoint_accs_seq']}")
    print(f"     process_rewards: {e['process_rewards']}")
