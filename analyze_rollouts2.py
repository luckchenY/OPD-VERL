import json, glob, os, re
from collections import defaultdict

ROOT = "/data/chenyang/opdswift/examples/train/test_result"

cases = [
    ("orig",  8192,  f"{ROOT}/DeepSeek-R1-Distill-Qwen-1.5B/20260624_230737/max_new_tokens_8192/native/20260624_230835"),
    ("orig",  16384, f"{ROOT}/DeepSeek-R1-Distill-Qwen-1.5B/20260625_015124/max_new_tokens_16384/native/20260625_015228"),
    ("orig",  32768, f"{ROOT}/DeepSeek-R1-Distill-Qwen-1.5B/20260625_015124/max_new_tokens_32768/native/20260625_015944"),
    ("opd1250", 8192,  f"{ROOT}/merged_hf/opd1250/max_new_tokens_8192/native/20260625_004520"),
    ("opd1250", 16384, f"{ROOT}/merged_hf/opd1250/max_new_tokens_16384/native/20260625_004936"),
    ("opd1250", 32768, f"{ROOT}/merged_hf/opd1250/max_new_tokens_32768/native/20260625_005718"),
    ("opdcons1050",      8192,  f"{ROOT}/merged_hf/opdconsistency1050/max_new_tokens_8192/native/20260630_002552"),
    ("opdcons1050",      16384, f"{ROOT}/merged_hf/opdconsistency1050/max_new_tokens_16384/native/20260630_005305"),
    ("opdcons1050long",  32768, f"{ROOT}/merged_hf/opdconsistency1050long/max_new_tokens_32768/native/20260630_025158"),
]

BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")

def load(path, ds):
    files = glob.glob(os.path.join(path, "reviews", "*", f"{ds}_default.jsonl"))
    recs = []
    with open(files[0]) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            r=json.loads(line)
            ac=r["messages"][1]["content"]
            if isinstance(ac,str):
                ac=[{"type":"text","content":ac}]
            reasoning=next((c.get("reasoning","") or "" for c in ac if isinstance(c,dict) and c.get("type")=="reasoning"),"")
            text=next((c.get("content","") or "" for c in ac if isinstance(c,dict) and c.get("type")=="text"),"")
            if not isinstance(text,str): text=json.dumps(text)
            score=r.get("sample_score",{}).get("score",{})
            acc=score.get("value",{}).get("acc",0)
            extracted=(score.get("extracted_prediction","") or "").strip()
            full=reasoning+"\n"+text
            has_box=bool(BOXED_RE.findall(full))
            has_box_text=bool(BOXED_RE.findall(text))
            # finished = emitted </think> (i.e. text part is the answer, non-empty & reasoning closed).
            # Proxy: reasoning non-empty AND text non-empty, OR boxed present in text.
            finished_think = bool(reasoning.strip()) and bool(text.strip())
            recs.append({
                "idx": r["index"],
                "group": r.get("sample_score",{}).get("group_id"),
                "acc": acc,
                "gen_chars": len(reasoning)+len(text),
                "rsn_chars": len(reasoning),
                "txt_chars": len(text),
                "has_box": has_box,
                "has_box_text": has_box_text,
                "extracted": extracted,
                "empty_extracted": (not extracted),
                "finished_think": finished_think,
                "rsn_empty": (not reasoning.strip()),
                "text": text,
                "reasoning": reasoning,
                "target": r.get("target",""),
            })
    return recs

def pct(seq,p):
    if not seq: return 0
    seq=sorted(seq)
    k=max(0,min(len(seq)-1,int(round(p/100*(len(seq)-1)))))
    return seq[k]

print("="*90)
print(" Per-cap comparison: orig vs train  (repeats=4 per problem, 30 problems per dataset)")
print("="*90)

for cap in [8192,16384,32768]:
    print(f"\n############ max_new_tokens = {cap} ############")
    for ds in ["aime24","aime25"]:
        print(f"\n----- {ds} -----")
        rows={}
        for label, c, path in cases:
            if c!=cap: continue
            rows[label]=load(path,ds)
        labels=[l for l,c,p in cases if c==cap]
        for label in labels:
            recs=rows[label]
            n=len(recs)
            if n==0:
                print(f"  [{label}] NO DATA")
                continue
            finished=[r for r in recs if r["has_box"]]
            truncated=[r for r in recs if not r["has_box"]]
            cond_acc = sum(r["acc"] for r in finished)/max(1,len(finished))
            by_g=defaultdict(list)
            for r in recs: by_g[r["group"]].append(r)
            prob_solved=sum(1 for g,rs in by_g.items() if any(x["acc"]>0.5 for x in rs))
            prob_finished=sum(1 for g,rs in by_g.items() if any(x["has_box"] for x in rs))
            print(f"  [{label}] n={n}")
            print(f"     overall acc        : {sum(r['acc'] for r in recs)/n:.3f}")
            print(f"     pass@4 (any correct): {prob_solved}/{len(by_g)} = {prob_solved/len(by_g):.3f}")
            print(f"     finished (boxed)    : {len(finished)}/{n} = {len(finished)/n:.3f}   (problems w/ any finished: {prob_finished}/{len(by_g)})")
            print(f"     acc | finished      : {cond_acc:.3f}  (n_fin={len(finished)})")
            print(f"     acc | truncated     : {sum(r['acc'] for r in truncated)/max(1,len(truncated)):.3f}  (n_trunc={len(truncated)})")
            print(f"     rsn_empty (no </think>): {sum(1 for r in recs if r['rsn_empty'])}/{n}={sum(1 for r in recs if r['rsn_empty'])/n:.3f}")
            print(f"     gen_chars: mean={sum(r['gen_chars'] for r in recs)/n:.0f} p50={pct([r['gen_chars'] for r in recs],50)} p90={pct([r['gen_chars'] for r in recs],90)} p99={pct([r['gen_chars'] for r in recs],99)}")
            # finished gen_chars vs truncated gen_chars
            fin_g=[r['gen_chars'] for r in finished]; trunc_g=[r['gen_chars'] for r in truncated]
            print(f"     finished gen_chars mean={sum(fin_g)/max(1,len(fin_g)):.0f}  | truncated gen_chars mean={sum(trunc_g)/max(1,len(trunc_g)):.0f}")
        # Per-problem delta: compare each non-orig model against orig
        if "orig" not in rows:
            continue
        og=rows["orig"]; og_g=defaultdict(list)
        for r in og: og_g[r["group"]].append(r)
        ogs={g for g,rs in og_g.items() if any(x["acc"]>0.5 for x in rs)}
        for label in labels:
            if label=="orig": continue
            tr=rows[label]; tr_g=defaultdict(list)
            for r in tr: tr_g[r["group"]].append(r)
            trs={g for g,rs in tr_g.items() if any(x["acc"]>0.5 for x in rs)}
            only_tr=trs-ogs; only_og=ogs-trs; both=ogs&trs
            print(f"  >> [{label} vs orig] PROBLEM-LEVEL DELTA: both={len(both)}, only {label}={len(only_tr)} {sorted(only_tr)}, only orig={len(only_og)} {sorted(only_og)}")
            for tag, probs in [(f"only-{label}", sorted(only_tr)), ("only-orig", sorted(only_og))]:
                if not probs: continue
                print(f"     -- {tag}:")
                for g in probs:
                    ors=og_g[g]; trs_g=tr_g[g]
                    ofin=sum(1 for r in ors if r["has_box"])
                    tfin=sum(1 for r in trs_g if r["has_box"])
                    og_mean=sum(r["gen_chars"] for r in ors)/len(ors)
                    tg_mean=sum(r["gen_chars"] for r in trs_g)/len(trs_g)
                    print(f"       prob {g}: orig fin {ofin}/{len(ors)} mean_gchars={og_mean:.0f} | {label} fin {tfin}/{len(trs_g)} mean_gchars={tg_mean:.0f}")
