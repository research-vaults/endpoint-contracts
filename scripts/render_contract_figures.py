#!/usr/bin/env python3
"""Render the two revised main-paper plots from frozen offline ledgers."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'ps.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})
meta={'CreationDate':None,'Creator':None,'Producer':None}
def load(p):return json.loads((ROOT/p).read_text())

d=load('results/contract_checks/donor_checks.json')['assignments']
keys=['length_nearest','current_text_tfidf','semantic_optimal']
labels=['Length nearest','Abstract TF–IDF','Abstract embedding']
fig,ax=plt.subplots(figsize=(5.5,1.65),layout='constrained')
for i,k in enumerate(keys):
    x=d[k]['delta'];m=x['mean'];lo,hi=x['ci95']
    ax.errorbar(m,2-i,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=['#546e7a','#0072b2','#009e73'][i])
ax.axvline(0,color='.65',lw=.8)
ax.set_yticks([2,1,0],labels);ax.set_xlabel('Mean true − donor recoverability (95% CI)')
ax.set_xlim(-.006,.18);ax.set_ylim(-.55,2.55)
fig.savefig(OUT/'contract_donor_effects.pdf',metadata=meta)
fig.savefig(OUT/'contract_donor_effects.png',dpi=180);plt.close(fig)

rows=load('results/openrouter_panel/panel_rows.json')
cache={r['key']:np.array(r['embedding']) for r in map(json.loads,(ROOT/'results/embedding_cache.jsonl').read_text().splitlines())}
def cos(a,b):return float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
fig,axes=plt.subplots(1,2,figsize=(5.5,2.2),layout='constrained',sharey=True)
stages=['S0_topic','S1_abstract','S3_methods_mid'];colors=['#8c8c8c','#0072b2','#009e73']
for st,c in zip(stages,colors):
    rs=[r for r in rows if r['stage']==st]
    axes[0].scatter([r['R_availability'] for r in rs],[r['score_lex_target_in_pred'] for r in rs],s=12,alpha=.65,color=c,label=st[:2])
rs=[r for r in rows if r['stage']=='S3_methods_mid']
xs=[cos(cache[r['item_id']+'|S3_methods_mid'],cache[r['item_id']+'|target']) for r in rs]
ys=[r['score_lex_target_in_pred'] for r in rs]
axes[1].scatter(xs,ys,s=12,alpha=.65,color=colors[2])
axes[0].set_xlabel('Lexical availability');axes[1].set_xlabel('Embedding availability (S3)')
axes[0].set_ylabel('Lexical endpoint score');axes[0].legend(frameon=False,fontsize=8)
axes[0].set_title('Same-family, pooled');axes[1].set_title('Different families, S3')
fig.savefig(OUT/'contract_score_associations.pdf',metadata=meta)
fig.savefig(OUT/'contract_score_associations.png',dpi=180);plt.close(fig)
# Record the plotted disjoint correlation rather than inheriting a prose number.
(ROOT/'results/contract_checks/figure_checks.json').write_text(json.dumps({'S3_n':len(xs),'S3_pearson':float(np.corrcoef(xs,ys)[0,1]),'scope':'same 76 retained S3 model-item cells; cached embeddings'},indent=2)+'\n')
