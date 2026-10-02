#!/usr/bin/env python3
"""Offline retrospective checks on retained inputs; seed and protocol are recorded below.

Never calls an API or overwrites historical results. All inference is conditional
on retained items/outputs, not repeated model generation or expert adjudication.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from recoverability.metrics import (bow_cosine, lexical_coverage, tokenize,
    rouge_l_f1, length_nearest_derangement, text_token_length)

OUT = ROOT / "results/contract_checks"
SEED = 20260928
INPUTS = ["data/disclosure_pilot/items.jsonl", "data/firebench/tasks.jsonl",
          "results/embedding_cache.jsonl", "results/openrouter_panel/panel_rows.json",
          "results/evidence_interventions.json", "results/oracle_closure.json"]


def read(path):
    p = ROOT / path
    return ([json.loads(x) for x in p.read_text().splitlines() if x.strip()]
            if p.suffix == ".jsonl" else json.loads(p.read_text()))


def save(name, obj):
    (OUT / name).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def interval(vals):
    a = np.asarray(vals, float)
    rng = np.random.default_rng(SEED)
    b = a[rng.integers(len(a), size=(5000, len(a)))].mean(axis=1)
    return {"n": len(a), "mean": float(a.mean()),
            "ci95": np.quantile(b, [.025, .975]).tolist(),
            "positive_fraction": float((a > 0).mean())}


def lcs_length(x, y):
    """Exact bit-vector LCS recurrence; checked against the original DP."""
    masks = {}
    for i, t in enumerate(y):
        masks[t] = masks.get(t, 0) | (1 << i)
    row = 0
    for t in x:
        hits = masks.get(t, 0) | row
        row = hits & ~(hits - ((row << 1) | 1))
    return row.bit_count()


def fast_r(a, b):
    x, y = tokenize(a), tokenize(b)
    if not x or not y:
        return 0.0
    sx, sy = set(x), set(y)
    return (len(sx & sy) / len(sy) + len(sx & sy) / len(sx | sy)
            + 2 * lcs_length(x, y) / (len(x) + len(y)) + bow_cosine(a, b)) / 4


def check_lcs(items):
    rng = np.random.default_rng(SEED)
    for _ in range(100):
        a = " ".join(rng.choice(["alpha", "beta", "gamma"], rng.integers(1, 40)))
        b = " ".join(rng.choice(["alpha", "beta", "gamma"], rng.integers(1, 40)))
        x, y = tokenize(a), tokenize(b)
        assert abs(rouge_l_f1(a, b) - 2*lcs_length(x, y)/(len(x)+len(y))) < 1e-12
    for it in items[:2]:
        a,b=it["stages"]["S3_methods_mid"],it["stages"]["target"]
        assert abs(rouge_l_f1(a,b)-2*lcs_length(tokenize(a),tokenize(b))/(len(tokenize(a))+len(tokenize(b)))) < 1e-12


def donor_checks(items):
    n = len(items)
    xs = [it["stages"]["S3_methods_mid"] for it in items]
    ys = [it["stages"]["target"] for it in items]
    lens = np.array([text_token_length(x) for x in xs])
    cache = {r["key"]: r["embedding"] for r in read(INPUTS[2])}
    v = np.array([cache[it["item_id"]+"|S1_abstract"] for it in items])
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    sim = v @ v.T
    old = np.array(length_nearest_derangement(xs))
    relative = np.abs(lens[None, :] - lens[:, None]) / np.maximum(lens[:, None], 1)
    allowed = relative <= .20
    allowed[np.arange(n), old] = True
    np.fill_diagonal(allowed, False)
    rng = np.random.default_rng(SEED)
    assignments = {"length_nearest": old}
    # Current-text topic check, added after the historical replay integrity stop.
    # This avoids relying solely on embeddings without per-text cache hashes.
    counts=[Counter(tokenize(it['stages']['S1_abstract'])) for it in items]
    vocab=sorted(set().union(*(set(c) for c in counts)))
    df=Counter(t for c in counts for t in c)
    tfidf=np.array([[c[t]*(np.log((1+n)/(1+df[t]))+1) for t in vocab] for c in counts])
    tfidf/=np.linalg.norm(tfidf,axis=1,keepdims=True)
    topic_sim=tfidf@tfidf.T
    _,tfidf_donor=linear_sum_assignment(np.where(allowed,-topic_sim,1e6))
    assignments['current_text_tfidf']=tfidf_donor
    for k in range(21):
        utility = sim.copy() if k == 0 else sim + .05 * rng.gumbel(size=(n,n))
        _, donor = linear_sum_assignment(np.where(allowed, -utility, 1e6))
        assert np.all(allowed[np.arange(n), donor]) and len(set(donor)) == n
        assignments["semantic_optimal" if k == 0 else f"semantic_jitter_{k:02d}"] = donor
    true = np.array([fast_r(x,y) for x,y in zip(xs,ys)])
    pair_cache = {}
    reports = {}
    for name, donor in assignments.items():
        swapped = []
        rows = []
        for i,j in enumerate(donor):
            key=(i,int(j))
            if key not in pair_cache:
                pair_cache[key] = fast_r(xs[j], ys[i])
            swapped.append(pair_cache[key])
            rows.append({"item_id":items[i]["item_id"],"donor_id":items[j]["item_id"],
                "true_R":float(true[i]),"donor_R":swapped[-1],
                "abstract_cosine":float(sim[i,j]),"relative_length_gap":float(relative[i,j]),
                "abstract_tfidf_cosine":float(topic_sim[i,j]),
                "input_bow_cosine":bow_cosine(xs[i],xs[j])})
        reports[name] = {"delta": interval(true-np.array(swapped)),
             "mean_true_R":float(true.mean()),"mean_donor_R":float(np.mean(swapped)),
             "mean_abstract_cosine":float(np.mean([r['abstract_cosine'] for r in rows])),
             "mean_abstract_tfidf_cosine":float(np.mean([r['abstract_tfidf_cosine'] for r in rows])),
             "mean_input_bow_cosine":float(np.mean([r['input_bow_cosine'] for r in rows])),
             "within_20_percent":float(np.mean([r['relative_length_gap']<=.2 for r in rows])),
             "rows":rows}
    old_report=read(INPUTS[4])
    replay = {'stored_delta':old_report['mean_delta_shuffle'],
              'recomputed_delta':reports['length_nearest']['delta']['mean'],
              'absolute_difference':abs(reports['length_nearest']['delta']['mean']-old_report['mean_delta_shuffle']),
              'status':'MISMATCH: retained text/code do not exactly regenerate historical ledger; new comparisons use one current input version'}
    attenuation = np.array([r['donor_R'] for r in reports['semantic_optimal']['rows']])-np.array([r['donor_R'] for r in reports['length_nearest']['rows']])
    save("donor_checks.json", {"protocol":"contract-checks-v1", "n":n,
         "historical_replay":replay,
         "target_used_for_matching":False,"matching_encoder":"cached text-embedding-3-small; first 6000 chars of S1; cache lacks text hashes, exact current-text binding unverified",
         "attenuation_length_minus_semantic_effect":interval(attenuation),
         "jitter_mean_effect_range":[min(reports[k]['delta']['mean'] for k in reports if 'jitter' in k),max(reports[k]['delta']['mean'] for k in reports if 'jitter' in k)],
         "assignments":reports})
    print('donors', {k:{a:v for a,v in d.items() if a!='rows'} for k,d in reports.items() if 'jitter' not in k},flush=True)


def pool_checks():
    tasks=read(INPUTS[1]); assert len(tasks)==35
    ys=[t['conclusion'] for t in tasks]
    rng=np.random.default_rng(SEED)
    output={"pool_n":35,"unique_conclusions":len(set(ys)),"candidate_origin":"all retained FIRE conclusions, including query gold; no alternative validity annotations","populations":{}}
    for field in ['instruction','research_question']:
        inds=[i for i,t in enumerate(tasks) if t.get(field)]
        pop={}
        for metric,fn in [('bow',bow_cosine),('coverage',lexical_coverage)]:
            scores=np.array([[fn(tasks[i][field][:12000] if field=='instruction' else tasks[i][field],y) for y in ys] for i in inds])
            rows=[]
            for r,i in enumerate(inds):
                rivals=[j for j in range(35) if j!=i]
                greater=sum(scores[r,j]>scores[r,i]+1e-12 for j in rivals)
                equal=sum(abs(scores[r,j]-scores[r,i])<=1e-12 for j in rivals)
                # Original stable-index rank, plus explicit tie ranges.
                rank=1+greater+sum(abs(scores[r,j]-scores[r,i])<=1e-12 and j<i for j in rivals)
                rows.append({'task_id':tasks[i]['task_id'],'rank':int(rank),'rank_best':int(1+greater),
                    'rank_worst':int(1+greater+equal),'margin':float(scores[r,i]-max(scores[r,rivals]))})
            sizes={}
            for size in [5,10,20,35]:
                random_mrr=[]; random_top=[]; hard_mrr=[]; hard_top=[]
                for r,i in enumerate(inds):
                    rivals=np.array([j for j in range(35) if j!=i])
                    hard=sorted(rivals,key=lambda j:(-scores[r,j],j))[:size-1]
                    def rank(pool):
                        return 1+sum((scores[r,j]>scores[r,i]+1e-12) or (abs(scores[r,j]-scores[r,i])<=1e-12 and j<i) for j in pool)
                    hr=rank(hard);hard_mrr.append(1/hr);hard_top.append(hr==1)
                    rr=[rank(rng.choice(rivals,size-1,replace=False)) for _ in range(200)]
                    random_mrr.append(float(np.mean(1/np.array(rr))));random_top.append(float(np.mean(np.array(rr)==1)))
                sizes[str(size)]={'random_mean_MRR':float(np.mean(random_mrr)),'random_mean_gold1':float(np.mean(random_top)),
                    'hard_MRR':float(np.mean(hard_mrr)),'hard_gold1':float(np.mean(hard_top))}
            pop[metric]={'n':len(inds),'MRR':float(np.mean([1/r['rank'] for r in rows])),
                'gold1':sum(r['rank']==1 for r in rows),'tied_query_count':sum(r['rank_best']!=r['rank_worst'] for r in rows),
                'mean_margin':float(np.mean([r['margin'] for r in rows])),'pool_sizes':sizes,'rows':rows}
        output['populations'][field]=pop
    assert output['populations']['instruction']['bow']['gold1']==31
    assert output['populations']['research_question']['bow']['gold1']==29
    # Known semantic contrast exposes exact preprocessing blindness.
    a='Treatment does improve survival.';b='Treatment does not improve survival.'
    assert tokenize(a)==tokenize(b)
    # f_c(t)=t+c*t*(t-1): all c fit observations at 0 and 1; only c=0 is affine.
    cases=[{'c':c,'observations':[0,1],'prediction_at_2':2+2*c,'affine_admissible':c==0,'quadratic_admissible':True} for c in [-1,0,1]]
    output['logical_controls']={'negation':{'positive':a,'negative':b,'tokens':tokenize(a),'R':fast_r(a,b),'bow':bow_cosine(a,b),'status':'analytic polarity contrast, not human scientific judgment'},
        'endpoint_family':{'definition':'f_c(t)=t+c*t*(t-1); exact observed f(0)=0,f(1)=1','cases':cases,'unique_under_affine':True,'nonunique_under_quadratic':True}}
    save('pool_checks.json',output)
    print('pools',{k:{m:{a:b for a,b in d.items() if a not in ['rows','pool_sizes']} for m,d in p.items()} for k,p in output['populations'].items()},flush=True)


def panel_checks():
    rows=read(INPUTS[3]);models=sorted({r['model'] for r in rows})
    stages=['S0_topic','S3_methods_mid']; by={}
    for r in rows:
        key=(r['item_id'],r['model'],r['stage']);assert key not in by
        by[key]=r['score_lex_target_in_pred']
    cohorts={'four_models':models,'gpt_qwen':[m for m in models if 'gpt' in m or 'qwen' in m]}
    out={}
    for name,ms in cohorts.items():
        ids=sorted(i for i in {r['item_id'] for r in rows} if all((i,m,s) in by for m in ms for s in stages))
        arr=np.array([[[by[i,m,s] for m in ms] for s in stages] for i in ids])
        records=[]
        for i,vals in zip(ids,arr):
            winners=[[ms[j] for j,v in enumerate(vs) if abs(v-max(vs))<=1e-12] for vs in vals]
            margins=[float(sorted(vs)[-1]-sorted(vs)[-2]) for vs in vals]
            records.append({'item_id':i,'winner_sets':winners,'margins':margins,
                            'disjoint_winner_sets':not bool(set(winners[0])&set(winners[1]))})
        summaries={}
        for si,s in enumerate(stages):
            means=arr[:,si,:].mean(axis=0);rank=np.argsort(-means)
            summaries[s]={'means':{m:interval(arr[:,si,j]) for j,m in enumerate(ms)},
                'ranking':[ms[j] for j in rank],
                'top_minus_runner_up':interval(arr[:,si,rank[0]]-arr[:,si,rank[1]]),
                'tied_items':sum(len(r['winner_sets'][si])>1 for r in records),
                'median_top_margin':float(np.median([r['margins'][si] for r in records]))}
        sensitivity={str(t):{'eligible':sum(min(r['margins'])>t for r in records),
                             'flips':sum(r['disjoint_winner_sets'] and min(r['margins'])>t for r in records)} for t in [0,.005,.01,.02]}
        out[name]={'n':len(ids),'models':ms,'stages':summaries,'margin_sensitivity':sensitivity,'rows':records,
                  'all_pairwise_stage_contrasts':{s:{ms[a]+' minus '+ms[b]:interval(arr[:,si,a]-arr[:,si,b]) for a in range(len(ms)) for b in range(a+1,len(ms))} for si,s in enumerate(stages)}}
    # Exactly reproduce original heterogeneous-panel max(), including first-row tie resolution.
    original=[]
    for i in sorted({r['item_id'] for r in rows}):
        rs=[[r for r in rows if r['item_id']==i and r['stage']==s] for s in stages]
        if min(map(len,rs))<2:continue
        w=[max(v,key=lambda r:r['score_lex_target_in_pred'])['model'] for v in rs]
        sets=[{r['model'] for r in v if abs(r['score_lex_target_in_pred']-max(z['score_lex_target_in_pred'] for z in v))<=1e-12} for v in rs]
        original.append({'item_id':i,'original_flip':w[0]!=w[1],'disjoint_winner_sets':not bool(sets[0]&sets[1]),'winner_sets':[sorted(x) for x in sets]})
    assert len(original)==20 and sum(r['original_flip'] for r in original)==13
    out['historical_heterogeneous']={'n':20,'original_flips':13,'tie_aware_flips':sum(r['disjoint_winner_sets'] for r in original),'rows':original}
    save('panel_checks.json',out)
    print('panel',{k:{a:v for a,v in d.items() if a not in ['rows','all_pairwise_stage_contrasts']} for k,d in out.items()},flush=True)


def select_sentences(inp, selection='central', target=None):
    parts=[p.strip() for p in re.split(r'(?<=[.!?])\s+',inp) if len(p.strip())>40]
    if not parts:return inp[:1500]
    if selection=='lead':return ' '.join(parts[:8])
    if selection=='gold':
        assert target is not None
        query=target
    else:
        assert target is None  # selection interface forbids gold for centrality.
        query=inp
    order=sorted(range(len(parts)),key=lambda j:-bow_cosine(parts[j],query))[:8]
    return ' '.join(parts[j] for j in order)


def extraction_checks(items):
    rows=[]
    for it in items:
        st=it['stages'];y=st['target']
        r={'item_id':it['item_id'],'topic_echo':lexical_coverage(st['S0_topic'],y),
           'S3_echo':lexical_coverage(st['S3_methods_mid'],y)}
        for stage in ['S3_methods_mid','S4_body']:
            for sel in ['gold','central','lead']:
                pred=select_sentences(st[stage],sel,y if sel=='gold' else None)
                r[stage+'_'+sel]=lexical_coverage(pred,y)
                r[stage+'_'+sel+'_chars']=len(pred)
        rows.append(r)
    weak=np.array([r['topic_echo'] for r in rows]);best=np.array([r['S3_echo'] for r in rows])
    rng=np.random.default_rng(SEED);bs=rng.integers(len(rows),size=(5000,len(rows)))
    out={'n':len(rows),'denominator':'mean S3 echo minus mean topic echo, lexical target-token coverage',
         'denominator_value':float(np.mean(best-weak)),'selectors':{},'rows':rows}
    for stage in ['S3_methods_mid','S4_body']:
        for sel in ['gold','central','lead']:
            key=stage+'_'+sel;v=np.array([r[key] for r in rows])
            closure=(v-weak).mean()/(best-weak).mean()
            ci=np.quantile((v-weak)[bs].mean(axis=1)/(best-weak)[bs].mean(axis=1),[.025,.975]).tolist()
            out['selectors'][key]={'score':interval(v),'closure':float(closure),'closure_ci95':ci,
                'mean_output_chars':float(np.mean([r[key+'_chars'] for r in rows])),
                'gold_used_for_selection':sel=='gold'}
    old=read(INPUTS[5])['closure_oracle_of_weak_to_best']
    out['historical_replay']={'stored_closure':old,'recomputed_closure':out['selectors']['S4_body_gold']['closure'],
        'absolute_difference':abs(out['selectors']['S4_body_gold']['closure']-old)}
    out['S4_gold_minus_central']=interval([r['S4_body_gold']-r['S4_body_central'] for r in rows])
    panel=read(INPUTS[3]); comparisons={}
    scores={r['item_id']:r['S3_methods_mid_central'] for r in rows}
    for m in sorted({r['model'] for r in panel}):
        rs=[r for r in panel if r['stage']=='S3_methods_mid' and r['model']==m]
        comparisons[m]={'n':len(rs),'central_minus_model':interval([scores[r['item_id']]-r['score_lex_target_in_pred'] for r in rs]),
            'caution':'same supplied S3 input and lexical outcome; output lengths not matched, no scientific-quality or LLM-superiority inference'}
    out['S3_panel_comparisons']=comparisons
    save('extraction_checks.json',out)
    print('extract',out['selectors'],flush=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--panel-only', action='store_true',
                        help='Recompute common-item model rankings using shipped numerical rows only.')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.panel_only:
        panel_checks()
        return
    missing = [p for p in INPUTS if not (ROOT/p).is_file()]
    if missing:
        parser.error('Full text-level replay needs separately obtained, licensed inputs: '
                     + ', '.join(missing) + '. See DATA_SOURCES.md; --panel-only runs offline.')
    items=[i for i in read(INPUTS[0]) if i.get('has_html_body')];assert len(items)==86
    manifest={'protocol':'contract-checks-v1','seed':SEED,'bootstrap':5000,'status':'retrospective exploratory',
              'inputs':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in INPUTS},
              'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'numpy':np.__version__}
    save('run_manifest.json',manifest)
    check_lcs(items);donor_checks(items);pool_checks();panel_checks();extraction_checks(items)


if __name__=='__main__':main()
