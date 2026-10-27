"""Paired metrics on immutable forecasts and identically frozen actuals."""
import numpy as np
import pandas as pd
from forecast.features import AREAS,TARGETS
from .ledger import dates,review_after

BANDS=[(0,6),(6,24),(24,48)]

def week_interval(rows,samples,seed):
    weeks=np.array([pd.Timestamp(r['origin']).strftime('%G-%V') for r in rows]);unique=np.unique(weeks)
    if len(unique)<6:return None
    totals=np.array([[sum(r['baselineError'] for r,w in zip(rows,weeks) if w==week),
                      sum(r['candidateError'] for r,w in zip(rows,weeks) if w==week)] for week in unique])
    rng=np.random.default_rng(seed);draw=totals[rng.integers(len(unique),size=(samples,len(unique)))].sum(1)
    good=draw[:,0]>0
    if not good.any():return None
    return np.quantile(100*(1-draw[good,1]/draw[good,0]),[.025,.975]).tolist()

def metrics(trial,at):
    p=trial['protocol'];at=pd.Timestamp(at);planned=dates(p);end=review_after(p)
    elapsed=int(sum(d+pd.Timedelta(hours=p['originHourUTC']+1)<=at for d in planned))
    complete_trial=at>=end;out={};observations=trial['observations']
    for candidate in ['primary',*p['secondaryCandidates']]:
        out[candidate]={}
        for area in AREAS:
            out[candidate][area]={}
            for target in TARGETS:
                rows=[];issued=0;partial_hours=0;eligible_records=0
                for record in trial['records']:
                    values=record['predictions'].get(candidate,{}).get(area,{}).get(target)
                    if values is None:continue
                    issued+=1
                    truth=[]
                    for t in record['times']:
                        observed=observations.get(pd.Timestamp(t).isoformat(),{}).get(area,{}).get(target,{})
                        valid=pd.Timestamp(t)
                        usable=(observed and pd.Timestamp(observed['capturedAt'])<=at and
                                valid+pd.Timedelta(hours=p['observationMinAgeHours'])<=at)
                        truth.append(observed['value'] if usable else None)
                    partial_hours+=sum(v is not None for v in truth)
                    if pd.Timestamp(record['times'][-1])+pd.Timedelta(hours=p['observationMinAgeHours'])<=at:eligible_records+=1
                    # The primary score uses whole 48h trajectories, not whichever
                    # hours happen to be easy or available early.
                    if any(v is None for v in truth):continue
                    y=np.asarray(truth);base=np.asarray(record['predictions']['baseline'][area][target]);pred=np.asarray(values)
                    old=np.abs(base-y);new=np.abs(pred-y);day=np.asarray(record['daylight'][area],bool)
                    rows.append(dict(origin=record['origin'],baselineError=float(old.sum()),candidateError=float(new.sum()),actual=float(y.sum()),
                        bands=[(float(old[lo:hi].sum()),float(new[lo:hi].sum())) for lo,hi in BANDS],
                        daylightHours=int(day.sum()),daylightBaselineError=float(old[day].sum()),daylightCandidateError=float(new[day].sum())))
                n=len(rows);old=sum(r['baselineError'] for r in rows);new=sum(r['candidateError'] for r in rows);actual=sum(r['actual'] for r in rows)
                weeks=len({pd.Timestamp(r['origin']).strftime('%G-%V') for r in rows})
                bands=[]
                for j,(lo,hi) in enumerate(BANDS):
                    b=sum(r['bands'][j][0] for r in rows);c=sum(r['bands'][j][1] for r in rows)
                    bands.append(dict(start=lo+1,end=hi,baselineMAE=b/(n*(hi-lo)) if n else None,
                                      candidateMAE=c/(n*(hi-lo)) if n else None,reductionPct=100*(1-c/b) if b>0 else None))
                reduction=100*(1-new/old) if old>0 else None
                sufficient=n>=p['minimumCompleteOrigins'] and weeks>=p['minimumCalendarWeeks'] and issued/len(planned)>=p['minimumIssueCoverage']
                meets=(sufficient and reduction is not None and reduction>=p['minimumMAEReductionPct'] and
                       all(b['reductionPct'] is not None and b['reductionPct']>=-p['maximumLeadBandRegressionPct'] for b in bands))
                day_hours=sum(r['daylightHours'] for r in rows)
                out[candidate][area][target]=dict(issuedOrigins=issued,completeOrigins=n,calendarWeeks=weeks,
                    observedForecastHours=partial_hours,eligibleMatureOrigins=eligible_records,
                    baselineMAEMW=old/(n*48) if n else None,candidateMAEMW=new/(n*48) if n else None,
                    candidateWAPEPct=100*new/actual if actual>0 else None,maeReductionPct=reduction,
                    daylightBaselineMAEMW=sum(r['daylightBaselineError'] for r in rows)/day_hours if day_hours and target=='solar' else None,
                    daylightCandidateMAEMW=sum(r['daylightCandidateError'] for r in rows)/day_hours if day_hours and target=='solar' else None,
                    leadBands=bands,descriptiveWeekBootstrap95Pct=week_interval(rows,p['bootstrapWeekSamples'],p['seed']),
                    decision='collecting' if not complete_trial else 'insufficient_data' if not sufficient else 'review_candidate' if meets else 'retain_baseline')
    return dict(version=p['version'],checkedAt=at.isoformat(),originStart=p['originStart'],originEndExclusive=p['originEndExclusive'],
        reviewAfter=end.isoformat(),scheduledOrigins=len(planned),elapsedIssueWindows=elapsed,issuedOrigins=len(trial['records']),
        missedIssueWindows=max(0,elapsed-sum(pd.Timestamp(r['origin'])+pd.Timedelta(hours=1)<=at for r in trial['records'])),
        status='review_ready' if complete_trial else 'collecting' if trial['records'] else 'waiting_for_first_issue',
        automaticPromotion=False,primaryRecipe=p['primaryRecipe'],metrics=out,
        lastAttempt=trial['attempts'][-1] if trial['attempts'] else None,
        lastIssue={k:trial['records'][-1].get(k) for k in ['origin','issuedAt','githubRunId','sourceErrors']} if trial['records'] else None,
        interpretation='Prospective paired errors on complete 48-hour trajectories. Overlapping dates are dependent. Weekly intervals are descriptive; no automatic model promotion.')
