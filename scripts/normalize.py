"""Normalize openly licensed observations without filling missing values.

Generation and exchange timestamps remain independent. Rolling snapshots merge
with a previous publication, so an upstream outage never erases valid history.
All power is MW; prices are EUR/MWh; dates are UTC epoch milliseconds.
"""
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path

from config import COUNTRIES, PRICE_ZONES
from fetch import write_json

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / '.cache'
DATA = ROOT / 'dist' / 'data'

GROUPS = {
    'wind_offshore': 'wind', 'wind_onshore': 'wind', 'wind': 'wind',
    'solar': 'solar', 'nuclear': 'nuclear', 'biomass': 'bio', 'geothermal': 'bio',
    'other_renewable': 'bio', 'other_renewables': 'bio',
    'hydro_run_of_river': 'hydro', 'hydro_run_of_river_and_poundage': 'hydro',
    'hydro_water_reservoir': 'hydro', 'hydro': 'hydro',
    'fossil_brown_coal_lignite': 'coal', 'fossil_hard_coal': 'coal',
    'fossil_gas': 'gas', 'fossil_coal_derived_gas': 'gas',
    'fossil_oil': 'other', 'fossil_oil_shale': 'other', 'fossil_peat': 'other',
    'others': 'other', 'other': 'other', 'waste': 'other',
    'hydro_pumped_storage': 'storage', 'battery': 'storage', 'battery_storage': 'storage',
}
AUXILIARY = {
    'cross_border_electricity_trading', 'load', 'residual_load',
    'renewable_share_of_load', 'renewable_share_of_generation',
    'hydro_pumped_storage_consumption', 'battery_storage_consumption', 'battery_consumption',
}
RENEWABLE = {'wind', 'solar', 'hydro', 'bio'}
NEIGHBOURS = {name.lower().replace(' ', '_'): code for code, (name, *_coords) in COUNTRIES.items()}
NEIGHBOURS.update({'czech_republic': 'cz', 'great_britain': 'uk', 'macedonia': 'mk',
                   'bosnia-herzegovina': 'ba', 'bosnia_herzegovina': 'ba',
                   'ukraine': 'ua', 'albania': 'al', 'moldova': 'md', 'turkey': 'tr',
                   'russia': 'ru', 'belarus': 'by', 'morocco': 'ma'})


def timestamp(value):
    value = value.replace('Z', '+00:00')
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError('Timestamp requires an explicit timezone')
    return int(parsed.timestamp() * 1000)


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def rounded(value):
    return round(value, 3) if number(value) else None


def complete_sum(values):
    values = list(values)
    return sum(values) if values and all(number(v) for v in values) else None


def normalize_power(raw, now_ms):
    if raw.get('unit') != 'MW':
        raise ValueError('Generation unit changed; expected MW')
    series = [s['id'] for s in raw['series']]
    unknown = set(series) - GROUPS.keys() - AUXILIARY
    if unknown:
        raise ValueError('Unclassified generation series: ' + ', '.join(sorted(unknown)))
    fuels = [s for s in series if s in GROUPS]
    rows = []
    for row in raw['data']:
        t = timestamp(row['timestamp'])
        if t > now_ms:
            continue
        values = row['values']
        if not any(number(values.get(s)) for s in fuels):
            continue
        grouped = {}
        missing = []
        for fuel in fuels:
            value = values.get(fuel)
            if not number(value) or value < 0:
                value = None
                missing.append(fuel)
            grouped.setdefault(GROUPS[fuel], []).append(value)
        g = {group: rounded(complete_sum(v)) for group, v in grouped.items()}
        primary = {k: v for k, v in g.items() if k != 'storage'}
        total = complete_sum(primary.values())
        renew = sum(v for k, v in primary.items() if k in RENEWABLE and number(v))
        low = renew + (primary.get('nuclear') or 0)
        rows.append({'t': t, 'g': g, 'total': rounded(total),
                     'renewable': round(100 * renew / total, 2) if total and total > 0 else None,
                     'lowCarbon': round(100 * low / total, 2) if total and total > 0 else None,
                     'load': rounded(values.get('load')), 'missing': missing})
    return rows


def normalize_flows(raw, now_ms):
    if raw.get('unit') != 'GW':
        raise ValueError('Exchange unit changed; expected GW')
    series = [s for s in raw['series'] if s['id'] != 'sum']
    rows = []
    for row in raw['data']:
        t = timestamp(row['timestamp'])
        if t > now_ms:
            continue
        values = row['values']
        if not any(number(values.get(s['id'])) for s in series):
            continue
        borders = [{'code': NEIGHBOURS.get(s['id']), 'name': s['name'],
                    'mw': rounded(values[s['id']] * 1000) if number(values.get(s['id'])) else None}
                   for s in series]
        net = complete_sum(b['mw'] for b in borders)
        rows.append({'t': t, 'borders': borders, 'net': rounded(net)})
    return rows


def read(path, fallback):
    return json.loads(path.read_text()) if path.exists() else fallback


def merge(old, new, cutoff):
    rows = {r['t']: r for r in old if r['t'] >= cutoff}
    rows.update({r['t']: r for r in new if r['t'] >= cutoff})
    return sorted(rows.values(), key=lambda r: r['t'])


def normalize_danish(now_ms, cutoff):
    output = read(DATA / 'denmark.json', {'areas': {}})
    production = read(CACHE / 'energinet' / 'ElectricityProdex5MinRealtime.json', {})
    carbon = read(CACHE / 'energinet' / 'CO2Emis.json', {})
    fields = ['ProductionLt100MW', 'ProductionGe100MW', 'OffshoreWindPower', 'OnshoreWindPower', 'SolarPower']
    borders = {'de': 'ExchangeGermany', 'nl': 'ExchangeNetherlands', 'uk': 'ExchangeGreatBritain',
               'no': 'ExchangeNorway', 'se': 'ExchangeSweden'}
    for area in ['DK1', 'DK2']:
        previous = output['areas'].get(area, {})
        rows = []
        for r in production.get('records', []):
            if r['PriceArea'] != area:
                continue
            t = timestamp(r['Minutes5UTC'] + 'Z')
            if t > now_ms:
                continue
            total = complete_sum(r.get(field) for field in fields)
            wind = complete_sum([r.get('OffshoreWindPower'), r.get('OnshoreWindPower')])
            # DK2 has external links only to Germany and Sweden. Null values in
            # fields for DK1-only links are structural absence, not missing DK2
            # observations; omit those links rather than inventing readings.
            connected = borders if area == 'DK1' else {code: borders[code] for code in ['de', 'se']}
            exchange = {code: rounded(r.get(field)) for code, field in connected.items()}
            # Great Belt is internal to Denmark. BornholmSE4 is already in ExchangeSweden.
            net = complete_sum(exchange.values())
            load = complete_sum([total, net, r.get('ExchangeGreatBelt')])
            rows.append({'t': t, 'total': rounded(total), 'wind': rounded(wind),
                         'solar': rounded(r.get('SolarPower')), 'load': rounded(load),
                         'borders': exchange, 'greatBelt': rounded(r.get('ExchangeGreatBelt'))})
        co2 = [{'t': timestamp(r['Minutes5UTC'] + 'Z'), 'value': rounded(r.get('CO2Emission'))}
               for r in carbon.get('records', []) if r['PriceArea'] == area
               and timestamp(r['Minutes5UTC'] + 'Z') <= now_ms]
        output['areas'][area] = {
            'power': merge(previous.get('power', []), rows, cutoff),
            'co2': merge(previous.get('co2', []), co2, cutoff),
        }
    output['source'] = 'Energinet / Energi Data Service'
    output['license'] = 'CC BY 4.0'
    output['method'] = 'Preliminary operational CO2 intensity of electricity consumption; not lifecycle CO2e.'
    write_json(DATA / 'denmark.json', output)


def main():
    now = datetime.now(timezone.utc)
    now_ms = int(now.timestamp() * 1000)
    cutoff = int((now - timedelta(days=7)).timestamp() * 1000)
    countries = {}
    errors = []
    for code, (name, lon, lat) in COUNTRIES.items():
        path = DATA / 'countries' / (code + '.json')
        previous = read(path, {})
        result = {'code': code, 'name': name, 'coordinates': [lon, lat],
                  'source': 'Fraunhofer ISE / Energy-Charts.info', 'license': 'CC BY 4.0'}
        for endpoint, key, normalizer in [('public_power', 'power', normalize_power), ('cbpf', 'flows', normalize_flows)]:
            raw = read(CACHE / endpoint / (code + '.json'), None)
            new = []
            if raw:
                try:
                    new = normalizer(raw, now_ms)
                except (ValueError, KeyError, TypeError) as error:
                    errors.append({'country': code, 'feed': endpoint, 'error': str(error)})
            result[key] = merge(previous.get(key, []), new, cutoff)
        result['prices'] = {}
        for zone, (country, label) in PRICE_ZONES.items():
            if country != code and not (code == 'lu' and zone == 'DE-LU'):
                continue
            raw = read(CACHE / 'price' / (zone + '.json'), None)
            old = previous.get('prices', {}).get(zone, {})
            new = []
            if raw and 'CC BY 4.0' in raw.get('license', '') and raw.get('unit') == 'EUR / MWh':
                new = [{'t': timestamp(r['timestamp']), 'value': rounded(r['values'].get('day_ahead_price'))}
                       for r in raw['data']]
            result['prices'][zone] = {'label': label, 'rows': merge(old.get('rows', []), new, cutoff),
                                     'intervalMinutes': raw.get('interval_minutes', 15) if raw else old.get('intervalMinutes', 15)}
        write_json(path, result)
        countries[code] = {k: result[k] for k in ['name', 'coordinates']}
        countries[code]['power'] = result['power'][-1] if result['power'] else None
        countries[code]['flows'] = result['flows'][-1] if result['flows'] else None
        # A small hourly map index avoids downloading every country's full history.
        hourly = {}
        for row in result['power']:
            hour = row['t'] // 3600000 * 3600000
            hourly.setdefault(hour, {})['power'] = {k: row[k] for k in ['t', 'total', 'renewable', 'lowCarbon']}
        for row in result['flows']:
            hour = row['t'] // 3600000 * 3600000
            hourly.setdefault(hour, {})['flows'] = {'t': row['t'], 'net': row['net']}
        countries[code]['history'] = [{'t': t, **v} for t, v in sorted(hourly.items())]
    normalize_danish(now_ms, cutoff)
    if not any(c['power'] for c in countries.values()):
        raise SystemExit('No valid generation snapshots; refusing to publish an empty atlas')
    write_json(DATA / 'index.json', {'schema': 1, 'generatedAt': now.isoformat(),
                                    'historyStart': cutoff, 'countries': countries,
                                    'coverage': sum(c['power'] is not None for c in countries.values()),
                                    'normalizationIssues': errors})
    for error in errors:
        print('Normalization issue:', error)
    print(f'Published {sum(bool(c["power"]) for c in countries.values())}/{len(countries)} generation feeds.')


if __name__ == '__main__':
    main()
