"""Forecast journeys, honest failure states, keyboard use and responsive layout."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlparse
from urllib.request import urlopen
from playwright.sync_api import sync_playwright, expect

ROOT=Path(__file__).resolve().parents[1]
URL=os.environ.get('GRID_TEST_URL','http://127.0.0.1:4322/')
OUT=ROOT/'test-results'

def check():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser=p.chromium.launch()
        page=browser.new_page(viewport={'width':1440,'height':1050},reduced_motion='reduce')
        errors=[];external=[]
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:external.append(request.url) if request.url.startswith('http') and urlparse(request.url).netloc!=urlparse(URL).netloc else None)
        page.goto(URL+'#forecasts',wait_until='networkidle')
        source=page.request.get(URL+'data/forecasts.json').json()
        assert source['latest'], 'Real forecast required for deployment'
        expect(page.locator('#forecast-stats>div')).to_have_count(3)
        assert page.locator('.forecast-predicted').get_attribute('d')
        page.locator('#forecasts').screenshot(path=str(OUT/'forecasts-desktop.png'))
        for area in ['DK2','DK1']:
            page.locator(f'[data-forecast-area="{area}"]').click()
            for target in ['solar','load','wind']:
                page.locator(f'[data-forecast-target="{target}"]').click()
                expect(page.locator(f'[data-forecast-target="{target}"]')).to_have_attribute('aria-pressed','true')
                assert page.locator('.forecast-predicted').get_attribute('d')
        page.locator('#forecast-chart').focus()
        page.keyboard.press('ArrowRight')
        expect(page.locator('#forecast-tooltip')).to_be_visible()
        page.keyboard.press('Escape')
        expect(page.locator('#forecast-tooltip')).to_be_hidden()
        with page.expect_download() as download:
            page.locator('#forecast-download').click()
        saved=json.loads(Path(download.value.path()).read_text())
        assert saved['issuedAt']==source['latest']['issuedAt']
        assert len(saved['times'])==48
        page.locator('[data-forecast-view="comparison"]').click()
        if source['comparison']:
            expect(page.locator('#forecast-timing')).to_contain_text('Comparing the forecast issued')
        else:
            expect(page.locator('#forecast-empty')).to_be_visible()
            expect(page.locator('#forecast-download')).to_be_disabled()
        page.locator('[data-forecast-view="outlook"]').click()
        for width in [390,768,1024]:
            page.set_viewport_size({'width':width,'height':844})
            page.wait_for_timeout(200)
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), f'Overflow at {width}px'
            page.locator('#forecasts').screenshot(path=str(OUT/f'forecasts-{width}.png'))
        page.locator('.forecast-methodology summary').click()
        expect(page.locator('.forecast-methodology')).to_have_attribute('open','')
        # Simulate source failure without altering any published forecasts.
        delayed=copy.deepcopy(source);delayed['status']='delayed'
        page.route('**/data/forecasts.json',lambda route:route.fulfill(json=delayed))
        page.reload(wait_until='networkidle')
        expect(page.locator('#forecast-status')).to_contain_text('Update delayed')
        assert page.locator('.forecast-predicted').get_attribute('d')
        page.unroute('**/data/forecasts.json')
        unavailable=copy.deepcopy(source);unavailable.update(latest=None,comparison=None,status='unavailable')
        page.route('**/data/forecasts.json',lambda route:route.fulfill(json=unavailable))
        page.reload(wait_until='networkidle')
        expect(page.locator('#forecast-empty')).to_be_visible()
        expect(page.locator('#forecast-download')).to_be_disabled()
        assert not errors, errors
        assert not external,external
        print(json.dumps({'forecastBrowser':'passed','viewports':[390,768,1024,1440],
            'checks':['six series','keyboard','download','comparison','delayed','unavailable','responsive','no external requests']}))
        browser.close()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--serve',action='store_true');args=parser.parse_args()
    server=None
    try:
        if args.serve:
            server=subprocess.Popen([sys.executable,'-m','http.server','4322','--bind','127.0.0.1','--directory',str(ROOT/'dist')],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            for _ in range(50):
                try:
                    with urlopen(URL,timeout=1):break
                except OSError:time.sleep(.1)
        check()
    finally:
        if server:server.terminate();server.wait(timeout=10)
