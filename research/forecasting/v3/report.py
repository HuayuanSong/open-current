"""Summarize all completed ablations without selecting from October winners."""
from research.forecasting.v3.common import *
import json
import numpy as np
import pandas as pd

def table(frame):
    rows=[list(frame.columns)]+frame.astype(str).values.tolist()
    return '\n'.join(['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']+
                     ['| '+' | '.join(row)+' |' for row in rows[1:]])

def baseline_noon(indices,areas):
    rule=json.loads((V2_CODE/'final_selection.json').read_text())
    p={f:np.load(V2/f'{f}_final.npz')['prediction'][indices] for f in FAMILIES}
    out=np.full_like(p['frozen'],np.nan)
    for s in rule['selection']:
        mask=np.asarray(areas)==s['area'];ti=TARGETS.index(s['target'])
        if s['selected']!='blend':out[mask,:,ti]=p[s['selected']][mask,:,ti];continue
        for (lo,hi),w in zip(BANDS,s['weights'],strict=True):
            out[mask,lo:hi,ti]=sum(weight*p[f][mask,lo:hi,ti] for f,weight in zip(FAMILIES,w,strict=True))
    return out

def comparisons():
    rows=[];origin_rows=[]
    for study in ['fleet','evening','regional','adapt']:
        path=CACHE/f'{study}-evaluate.npz' if study!='adapt' else CACHE/'adapt-evaluation.npz'
        data=dict(np.load(path))
        m=pd.read_csv(HERE/f'{study}-evaluate-origins.csv' if study!='adapt' else HERE/'adapt-origins.csv')
        if study=='fleet':base=baseline_noon(m.original_index.to_numpy(),m.area)
        elif study=='adapt':base=data['fixed']
        else:
            bm=pd.read_csv(HERE/'baseline-evening-origins.csv');bp=np.load(CACHE/'baseline-evening.npz')['prediction']
            lookup={(pd.Timestamp(r.origin),r.area):i for i,r in enumerate(bm.itertuples())}
            base=bp[[lookup[(pd.Timestamp(r.origin),r.area)] for r in m.itertuples()]]
        for name,p in data.items():
            if name in ['actual','fixed'] or name.startswith('correction_') or name.startswith('gate_'):continue
            for area in AREAS:
                mask=m.area.eq(area).to_numpy()
                for ti,target in enumerate(TARGETS):
                    valid=mask&np.isfinite(p[:,:,ti]).all(1)
                    if not valid.any():continue
                    y=data['actual'][valid,:,ti];new=np.abs(p[valid,:,ti]-y);old=np.abs(base[valid,:,ti]-y)
                    per_new=new.mean(1);per_old=old.mean(1)
                    rows.append(dict(study=study,area=area,target=target,candidate=name,origins=int(valid.sum()),
                        baseline_mae_mw=float(old.mean()),candidate_mae_mw=float(new.mean()),
                        improvement_pct=float((1-new.sum()/old.sum())*100),origins_improved=int((per_new<per_old).sum())))
                    for origin,n,o in zip(m.loc[valid,'origin'],per_new,per_old,strict=True):
                        origin_rows.append(dict(study=study,area=area,target=target,candidate=name,origin=origin,
                                                baseline_mae_mw=float(o),candidate_mae_mw=float(n)))
    pd.DataFrame(rows).to_csv(HERE/'versus-deployed.csv',index=False)
    pd.DataFrame(origin_rows).to_csv(HERE/'per-origin-errors.csv',index=False)
    return pd.DataFrame(rows)

def main():
    frame=comparisons();audit=json.loads((HERE/'data-audit.json').read_text());fleet=json.loads((HERE/'fleet-audit.json').read_text())
    from research.forecasting.v3.weather import read
    availability={};missing=[]
    for kind in ['fleet','fresh','regional']:
        paths=sorted((CACHE/'weather'/kind).glob('*.json.gz'));count=0
        for path in paths:
            record=read(path)
            if isinstance(record['data'],list):count+=1
            else:missing.append(dict(source=kind,run=record['run'],reason=record['data'].get('reason')))
        availability[kind]=dict(cached_runs=len(paths),available_runs=count)
    write_json(HERE/'source-availability.json',dict(counts=availability,missing=missing))
    sections=[]
    for study in ['fleet','evening','regional','adapt']:
        metrics=pd.read_csv(HERE/f'{study}-evaluate.csv')
        metrics=metrics[(metrics.lead_start==1)&(metrics.lead_end==48)]
        if study=='adapt':
            metrics=metrics[metrics.candidate.isin(['fixed','selected_correction','selected_gate','combined'])]
            cols=['area','target','candidate','origins','mae_mw','improvement_pct']
        else:cols=['area','target','variant','origins','mae_mw','improvement_vs_control_pct']
        metrics=metrics[cols].copy()
        for col in metrics.select_dtypes('float').columns:metrics[col]=metrics[col].map(lambda x:f'{x:.2f}')
        sections.append(f'## {study}\n\n'+table(metrics))
    relative=frame[['study','area','target','candidate','origins','baseline_mae_mw','candidate_mae_mw','improvement_pct','origins_improved']].copy()
    for col in relative.select_dtypes('float').columns:relative[col]=relative[col].map(lambda x:f'{x:.2f}')
    runtime=[]
    for p in HERE.glob('*-runtime.json'):
        records=json.loads(p.read_text())
        runtime.append({'stage':p.stem,'seconds':round(sum(r.get('train_seconds',r.get('seconds',0))+r.get('inference_seconds',0) for r in records),2)})
    text=f'''# Forecast inputs and adaptation: local results

All four experiments were implemented and run with real public data. The
deployed v2 ensemble is preserved. These are retrospective comparisons:
June–October 2026 was already inspected during previous research. Hyperparameter
selection uses pre-October labels, but October is **not a fresh independent
holdout** for the project. Overlapping 48-hour windows are not independent.

Read [protocol](PROTOCOL.md), [implementation notes](IMPLEMENTATION_NOTES.md),
and [reproduction instructions](README.md). Positive improvement means lower
MAE; negative means worse. WAPE and lead-band results are in the CSVs. MAE is
in MW and is not a percentage accuracy score.

## Findings

- Capacity-aware wind inputs reduced noon DK1 MAE from **345.69 to 327.97 MW**
  versus the deployed ensemble (5.1%); DK2 became worse (5.9%). This is not a
  universal benefit from capacity weighting.
- Newer ECMWF weather plus capacity features at evening origins reduced wind
  error by **8.6% in DK1 and 7.5% in DK2**. The same newer-weather pipeline
  reduced demand error by **6.1% and 2.2%**. Capacity inputs only affect wind.
- Combined ECMWF/DMI inputs reduced evening solar error by **13.3% and 10.5%**
  versus the deployed ensemble. Against matched shorter-history ECMWF tree
  controls, the reductions are smaller; use the regional table below to isolate
  the additional weather source. DMI alone performed poorly on wind.
- Online correction reduced demand error by about **0.6%** in both areas but
  worsened DK2 wind. Adaptive weights gave small mixed changes, including a
  1.5% DK2 wind improvement and worse DK2 solar/demand. Neither is a broad win.

These are observations about tested candidates, not an independently verified
selection of the best future model. **Prioritize weather-input challengers for
a prospective trial; keep the current public model until that trial supplies
new evidence.** Freeze the candidate specification before those outcomes
arrive. Selecting different winners from these October results would require
another independent evaluation period.

## Data and comparison scope

- Turbine register: {fleet['usable_rows']:,} usable records; {fleet['duplicate_technical_rows']}
  duplicate technical records removed, {fleet['excluded_rows']:,} excluded for
  unavailable/invalid required fields ({fleet['excluded_known_mw']:.2f} MW of
  known capacity, including retired records). This is a revised September
  register, not an archive of what was published on each historical date.
  Municipality-zero North Sea offshore records are explicitly mapped to DK1.
  Connection dates do not establish when a registry update became public;
  this source-vintage limitation requires prospective validation.
- Geography uses only pre-January-2025 active turbines to choose six fixed
  representatives. Later capacity is assigned to the same clusters. Land and
  offshore wind stay separate. No production columns are used; solar/demand
  retain their original location sampling.
- Matched electricity eligibility: {audit['datasets']['12']['origins_per_area']['DK1']} noon dates and
  {audit['datasets']['18']['origins_per_area']['DK1']} evening dates per area across the full dataset.
  The regional comparison has {audit['datasets']['18']['regional_complete_rows']//2} complete dates per area.
  Input exclusions are in weather-exclusions.csv; unavailable archived runs
  are recorded separately in source-availability.json.
- Evening experiments compare 00 and 06 weather runs at 18:15 issuance. A
  conservative 12h15 publication assumption is used in replay. A separate
  tested live selector uses actual availability metadata plus ten minutes.
- DMI comparisons have shorter matched histories from April 2026 onward.
  The combined model receives both ECMWF and DMI fields. Missing DMI runs are
  not replaced with later runs or observed weather. DMI covers the first 47
  hours; the terminal hour uses the already-available same-issue ECMWF forecast
  and an explicit source indicator. See [horizon addendum](REGIONAL_ADDENDUM.md).
- Corrections use only matured earlier labels; overlapping errors are
  deduplicated by target timestamp and lead band. Daily historical issues are
  a limited test of a correction intended eventually for hourly operation.
  Base-family settings were already selected in v2 using the development
  periods, so this is not fully nested validation of the entire model stack.
  Results are conditional on complete input histories; missing-source periods
  may differ from the dates retained in the comparison.

{(chr(10)+chr(10)).join(sections)}

## Comparison with the actual deployed ensemble

Weather tables above isolate input changes against matched refitted tree
controls. This table compares them with the deployed nine-family ensemble,
which is a stronger baseline. Evening ensemble predictions were recomputed
using the real deployed artifacts and matching evening histories. The deployed
models were trained at noon origins; evening tree controls were refitted at
evening origins. This is a practical pipeline comparison, not an architecture
comparison. No winner was selected using these October scores.

{table(relative)}

## Measured local CPU cost

{table(pd.DataFrame(runtime))}

Training ran on the local Mac, sometimes concurrently; figures are elapsed
times, not isolated hardware benchmarks. Downloads and environment setup are
separate. New tree and gate inference requires CPU only. All model weights,
source hashes, selections and origin-level errors are retained locally.

## Release decision

No automatic promotion from these reused historical periods. The prospective
step is to freeze a challenger and record its issued predictions alongside
the deployed model, then score newly matured outcomes. A point-forecast change
also requires new range calibration and a renewed explanation audit. Existing
prediction ranges must not be attached to a different point model unchanged.
'''
    (HERE/'RESULTS.md').write_text(text)
    plot(frame)
    paths=sorted((CACHE/'weather-models').glob('*'))+sorted((CACHE/'adapt-models').glob('*'))
    write_json(HERE/'artifacts.json',dict(files={str(p.relative_to(CACHE)):dict(bytes=p.stat().st_size,sha256=digest(p)) for p in paths},
        protocol_sha256=digest(HERE/'PROTOCOL.md'),feature_names_sha256=digest(HERE/'feature-names.json'),
        regional_addendum_sha256=digest(HERE/'REGIONAL_ADDENDUM.md'),
        source_sha256={p.name:digest(p) for p in HERE.glob('*.py')},
        weather_manifest_sha256=digest(HERE/'weather-manifest.json'),
        status='research challengers; not deployed; no independent prospective score yet'))

def plot(frame):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(12.5,7.5),layout='constrained',sharey=True)
    keep=frame[(frame.study.eq('adapt')&frame.candidate.isin(['selected_correction','selected_gate','combined']))|
               (frame.study.eq('fleet')&frame.candidate.eq('fleet'))|
               (frame.study.eq('evening')&frame.candidate.eq('fresh_fleet'))|
               (frame.study.eq('regional')&frame.candidate.eq('multi_weather'))]
    order=['fleet','fresh_fleet','multi_weather','selected_correction','selected_gate','combined']
    labels={'selected_correction':'Error correction · noon','selected_gate':'Adaptive weights · noon',
            'combined':'Both adaptations · noon','fleet':'Fleet inputs · noon',
            'fresh_fleet':'Newer + fleet · evening','multi_weather':'ECMWF + DMI · evening'}
    for ai,area in enumerate(AREAS):
        for ti,target in enumerate(TARGETS):
            ax=axes[ai,ti];part=keep[keep.area.eq(area)&keep.target.eq(target)]
            values=[float(part.loc[part.candidate.eq(k),'improvement_pct'].iloc[0]) if part.candidate.eq(k).any() else np.nan for k in order]
            ax.barh(np.arange(len(order)),values,
                    color=['#087f72' if v>=0 else '#bb6948' for v in values])
            ax.set_yticks(np.arange(len(order)),[labels[k] for k in order]);ax.set_ylim(5.6,-.6)
            ax.axvline(0,color='#556066',lw=.8);ax.set_title(f'{area} · {"demand" if target=="load" else target}')
            if ti:ax.tick_params(axis='y',labelleft=False)
            for i,v in enumerate(values):
                if not np.isfinite(v):ax.text(0,i,'Not tested',va='center',fontsize=8,color='#7a8589')
    fig.supxlabel('MAE reduction versus the deployed ensemble (%) — positive values mean lower error')
    fig.suptitle('Exploratory comparison on 13 previously inspected October dates',fontsize=14)
    fig.savefig(HERE/'comparison.png',dpi=160);fig.savefig(HERE/'comparison.svg');plt.close(fig)

if __name__=='__main__':main()
