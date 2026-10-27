"""Capacity-only fleet geography; no production data enters clustering."""
from research.forecasting.v3.common import CACHE, HERE, digest, write_json
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans

URL = 'https://ens.dk/media/8748/download'

def number(series):
    return pd.to_numeric(series.astype(str).str.replace(',', '.', regex=False), errors='coerce')

def active(frame, at):
    at = pd.Timestamp(at)
    return (frame.connected <= at) & (frame.disconnected.isna() | (frame.disconnected > at))

def bidding_area(municipality,kind,longitude,latitude):
    if not np.isfinite(municipality):return None
    if municipality==400:return 'DK2'
    if municipality==0:
        if not np.isfinite(longitude) or not np.isfinite(latitude):return None
        # Offshore installations need not belong to a municipality. North Sea
        # Danish turbines west of 9E connect to western Denmark, not DK2.
        if kind=='sea' and 3<longitude<9 and 54<latitude<59:return 'DK1'
        raise ValueError('Unmapped offshore/municipality-zero record; inspect grid connection')
    return 'DK1' if municipality>400 else 'DK2'

def main():
    path = CACHE / 'wind-register.xlsx'
    if not path.exists():
        import requests
        response = requests.get(URL, timeout=60); response.raise_for_status()
        path.write_bytes(response.content)
    raw = pd.read_excel(path, header=2, usecols=range(16), dtype=str)
    raw.columns = raw.columns.str.strip()
    # The published workbook reverses these two column headings. Validate values
    # before correcting; fail rather than silently accepting a changed schema.
    placement = raw['Kommune']
    municipality = raw['Type af placering']
    assert set(placement.dropna().unique()) <= {'LAND','HAV'}, 'Register schema changed'
    frame = pd.DataFrame({
        'id':raw.iloc[:,0], 'connected':pd.to_datetime(raw.iloc[:,2], utc=True, errors='coerce'),
        'disconnected':pd.to_datetime(raw.iloc[:,3], utc=True, errors='coerce'),
        'mw':number(raw['Kapacitet (kW)'])/1000,
        'hub_m':number(raw['Navhøjde (m)']), 'kind':placement.map({'LAND':'land','HAV':'sea'}),
        'municipality':pd.to_numeric(municipality.str.extract(r'^(\d+)')[0], errors='coerce'),
        'x':number(raw['UTM x-koordinat']), 'y':number(raw['UTM y-koordinat'])})
    frame = frame.drop_duplicates().reset_index(drop=True)
    frame['area'] = np.where(frame.municipality >= 400, 'DK1','DK2')
    # Municipality 400 is Bornholm (DK2), despite the numerical west/east split.
    frame.loc[frame.municipality.eq(400), 'area'] = 'DK2'
    for bornholm, crs in [(False,25832),(True,25833)]:
        mask = frame.municipality.eq(400).eq(bornholm)
        lon, lat = Transformer.from_crs(crs,4326,always_xy=True).transform(frame.loc[mask,'x'].to_numpy(),frame.loc[mask,'y'].to_numpy())
        frame.loc[mask,'longitude'] = lon; frame.loc[mask,'latitude'] = lat
    frame['area']=[bidding_area(r.municipality,r.kind,r.longitude,r.latitude) for r in frame.itertuples()]
    valid = (frame.mw.gt(0) & frame.connected.notna() & frame.kind.notna() &
             frame.municipality.notna() & frame.area.notna() & frame.longitude.between(3,16) & frame.latitude.between(54,58.5))
    audit = {'source':URL,'sha256':digest(path),'rows':len(raw),'usable_rows':int(valid.sum()),
             'duplicate_technical_rows':len(raw)-len(frame),
             'excluded_rows':int((~valid).sum()),'excluded_known_mw':float(frame.loc[~valid,'mw'].sum()),
             'header_repair':'Kommune and Type af placering are reversed in the source',
             'municipality_zero_rows':int(frame.municipality.eq(0).sum()),
             'mapped_north_sea_rows':int((frame.municipality.eq(0)&frame.area.eq('DK1')).sum()),
             'offshore_area_rule':'Municipality 0, North Sea west of 9E -> DK1; any other code-0 record fails for inspection',
             'vintage':'September 2026 revised register; historical release vintages unavailable'}
    frame = frame.loc[valid].copy().reset_index(drop=True)
    assert not frame.id.duplicated().any(), 'Conflicting usable records for one turbine'
    points=[]
    for area, kind, k in [('DK1','land',2),('DK1','sea',1),('DK2','land',1),('DK2','sea',2)]:
        group = frame.area.eq(area) & frame.kind.eq(kind)
        train = group & active(frame,pd.Timestamp('2025-01-01',tz='UTC'))
        coords = frame[['longitude','latitude']].to_numpy()*[np.cos(np.radians(56)),1]
        model = KMeans(n_clusters=k,random_state=42,n_init=10).fit(coords[train],sample_weight=frame.loc[train,'mw'])
        assignment = model.predict(coords[group])
        for c in range(k):
            # Representative is an actual pre-2025 turbine, not a land/sea centroid.
            candidates = np.flatnonzero(train.to_numpy())
            nearest = candidates[np.linalg.norm(coords[candidates]-model.cluster_centers_[c],axis=1).argmin()]
            p=frame.iloc[nearest]; point_id=f'{area}_{kind}_{c}'
            points.append(dict(id=point_id,area=area,kind=kind,latitude=float(p.latitude),longitude=float(p.longitude)))
            rows=frame.index[group][assignment==c]; frame.loc[rows,'cluster']=point_id
    assert frame.cluster.notna().all()
    frame.to_parquet(CACHE/'fleet.parquet',index=False)
    audit['active_mw']={str(day):frame.loc[active(frame,pd.Timestamp(day,tz='UTC'))].groupby(['area','kind']).mw.sum().to_dict()
                        for day in ['2025-01-01','2026-09-30']}
    audit['active_mw']={d:{'/'.join(k):v for k,v in x.items()} for d,x in audit['active_mw'].items()}
    write_json(HERE/'fleet-audit.json',audit); write_json(HERE/'weather-points.json',points)
    print(audit,flush=True); print(points,flush=True)

if __name__=='__main__': main()
