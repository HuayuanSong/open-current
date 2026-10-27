"""Package immutable trial weights, fixed fleet capacity and parity fixtures."""
import json
import shutil
import tarfile
import numpy as np
import pandas as pd
from .artifacts import ROOT,digest
from .score import VARIANTS,filename
from forecast.features import AREAS,TARGETS

def main():
    protocol=json.loads((ROOT/'forecast/trial/protocol.json').read_text());version=protocol['version']
    source=ROOT/'.cache/forecasting/v3';research=ROOT/'research/forecasting/v3'
    directory=ROOT/'.cache/forecast-serving/trial-models'/version;directory.mkdir(parents=True,exist_ok=True)
    def copy(path,name):
        dest=directory/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,dest)
    expected_hashes=json.loads((research/'artifacts.json').read_text())['files']
    for name in sorted({filename(v,a,t) for v in VARIANTS for a in AREAS for t in TARGETS}):
        path=source/'weather-models'/name
        if digest(path)!=expected_hashes['weather-models/'+name]['sha256']:raise ValueError('Research weights changed')
        copy(path,'models/'+name)
    for name in ['weather-points.json','feature-names.json','fleet-audit.json']:
        copy(research/name,name)
    fleet=pd.read_parquet(source/'fleet.parquet');at=pd.Timestamp(protocol['frozenFleetAsOf'])
    live=fleet[(fleet.connected<=at)&(fleet.disconnected.isna()|(fleet.disconnected>at))]
    capacity=live.groupby('cluster').mw.sum().to_dict()
    (directory/'capacity.json').write_text(json.dumps(capacity,indent=2)+'\n')
    a=dict(np.load(source/'data-18.npz'));m=pd.read_csv(source/'samples-18.csv')
    origins=pd.to_datetime(m.origin,utc=True);ids=np.flatnonzero(origins>=pd.Timestamp('2026-10-01',tz='UTC'))[[0,1,-2,-1]]
    fixture={k:v[ids] for k,v in a.items() if k!='y'};fixture['area_ids']=m.loc[ids,'area_id'].to_numpy()
    np.savez_compressed(directory/'fixture.npz',**fixture)
    references={}
    for variant in VARIANTS:
        study='regional' if variant=='regional_weather' else 'evening'
        field={'evening_control':'control','newer_weather':'fresh','fleet_weather':'fresh_fleet','regional_weather':'multi_weather'}[variant]
        meta=pd.read_csv(research/f'{study}-evaluate-origins.csv')
        lookup={(pd.Timestamp(r.origin),r.area):i for i,r in enumerate(meta.itertuples())}
        indices=[lookup[(pd.Timestamp(m.iloc[i].origin),m.iloc[i].area)] for i in ids]
        references[variant]=np.load(source/f'{study}-evaluate.npz')[field][indices]
    np.savez_compressed(directory/'expected.npz',**references)
    copy(ROOT/'forecast/trial/protocol.json','protocol.json')
    copy(ROOT/'LICENSE','LICENSE')
    (directory/'NOTICE.txt').write_text('Open Current prospective weather trial. Project and trained LightGBM models: MIT.\nWeather: Open-Meteo, ECMWF and DMI; https://open-meteo.com/en/licence (CC BY 4.0).\nFleet: Danish Energy Agency, September 2026 revised register, https://ens.dk/media/8748/download\nElectricity: Energinet Energi Data Service. Research and limitations: https://github.com/HuayuanSong/open-current/tree/main/research/forecasting/v3\n')
    files={str(p.relative_to(directory)):digest(p) for p in directory.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (directory/'manifest.json').write_text(json.dumps(dict(version=version,protocolSha256=digest(ROOT/'forecast/trial/protocol.json'),files=files),indent=2)+'\n')
    archive=ROOT/'.cache/forecasting/release'/f'{version}.tar.gz'
    with tarfile.open(archive,'w:gz') as dest:dest.add(directory,arcname=version)
    config=dict(version=version,url=f'https://github.com/HuayuanSong/open-current/releases/download/{version}/{archive.name}',sha256=digest(archive),bytes=archive.stat().st_size)
    (ROOT/'forecast/trial/model-release.json').write_text(json.dumps(config,indent=2)+'\n')
    print(config)

if __name__=='__main__':main()
