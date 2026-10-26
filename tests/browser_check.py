"""User journeys and responsive checks against real bundled data."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'test-results'
URL = os.environ.get('GRID_TEST_URL', 'http://127.0.0.1:4322/')


def check():
    OUT.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(viewport={'width': 1440, 'height': 1050}, reduced_motion='reduce')
        page = context.new_page()
        errors, external = [], []
        page.on('pageerror', lambda error: errors.append(str(error)))
        page.on('request', lambda request: external.append(request.url) if request.url.startswith('http') and urlparse(request.url).netloc != urlparse(URL).netloc else None)
        page.goto(URL, wait_until='networkidle')
        expect(page.locator('#country-select')).to_have_value('dk')
        expect(page.locator('#detail-value')).not_to_have_text('—')
        expect(page.locator('.country-shape')).to_have_count(67)
        assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), 'Desktop overflow'
        page.screenshot(path=str(OUT / 'desktop.png'), full_page=True)

        page.locator('[data-panel="flows"]').click()
        assert page.locator('#flows-list .flow-row').count() > 0
        assert page.locator('.flow-path').count() > 0
        page.locator('#show-flows').uncheck()
        expect(page.locator('.flow-path')).to_have_count(0)
        page.locator('#show-flows').check()
        page.locator('[data-metric="net"]').click()
        expect(page.locator('#detail-metric-label')).to_have_text('Net imports')
        assert 'GW' in page.locator('#detail-value').inner_text()
        page.locator('[data-metric="renewable"]').click()
        expect(page.locator('#detail-metric-label')).to_have_text('Renewable generation')

        # Wheel zoom produces fractional scales; dragging changes both offsets.
        box = page.locator('#map').bounding_box()
        x,y = box['x']+box['width']*.5, box['y']+box['height']*.55
        page.mouse.move(x,y);page.mouse.wheel(0,-110)
        page.wait_for_function('document.querySelector("#map").__zoom.k > 1')
        before=page.evaluate('({...document.querySelector("#map").__zoom})')
        assert before['k'] != round(before['k']), 'Zoom was quantized'
        page.mouse.down();page.mouse.move(x+47,y+31,steps=8);page.mouse.up()
        after=page.evaluate('({...document.querySelector("#map").__zoom})')
        assert abs(after['x']-before['x'])>20 and abs(after['y']-before['y'])>15, 'Pan failed'
        page.locator('#zoom-reset').click()
        page.wait_for_function('Math.abs(document.querySelector("#map").__zoom.k - 1) < .01')

        page.locator('#country-select').select_option('fr')
        expect(page.locator('#chart-country')).to_have_text('/ France')
        page.wait_for_function('document.querySelector("#country-time").textContent.includes("UTC")')
        expect(page.locator('#denmark-section')).to_be_hidden()
        page.locator('[data-chart="share"]').click()
        expect(page.locator('.chart-line')).to_have_count(2)

        # Historical hour should affect map and detail, and survive reload.
        page.locator('#time-slider').evaluate('(el) => {el.value=80;el.dispatchEvent(new Event("input",{bubbles:true}));}')
        assert 'time=' in page.url
        assert 'Latest available' not in page.locator('#time-label').inner_text()
        selected_time=page.locator('#time-label').inner_text()
        page.reload(wait_until='networkidle')
        expect(page.locator('#country-select')).to_have_value('fr')
        expect(page.locator('#time-label')).to_have_text(selected_time)
        page.locator('#play-time').click()
        page.wait_for_function('(previous) => document.querySelector("#time-label").textContent !== previous',arg=selected_time)
        page.locator('#play-time').click()
        page.locator('#latest-time').click()
        expect(page.locator('#time-label')).to_have_text('Latest available')

        page.locator('#country-select').select_option('dk')
        page.wait_for_function('document.querySelector("#price-zone").options.length===2')
        page.locator('#price-zone').select_option('DK2')
        page.locator('[data-chart="price"]').click()
        expect(page.locator('#chart-caption')).to_contain_text('DK2')
        assert page.locator('.chart-line').count()==1
        page.reload(wait_until='networkidle')
        expect(page.locator('#price-zone')).to_have_value('DK2')
        expect(page.locator('#chart-caption')).to_contain_text('DK2')
        with page.expect_download() as download:
            page.locator('#download-data').click()
        assert download.value.suggested_filename=='grid-in-motion-dk.json'

        page.locator('[data-about]').first.click()
        expect(page.locator('#about-dialog')).to_be_visible()
        page.keyboard.press('Escape')
        expect(page.locator('#about-dialog')).to_be_hidden()

        page.locator('#show-all').click()
        expect(page.locator('.ranking-row')).to_have_count(34)
        page.locator('#ranking-metric').select_option('net')
        page.locator('#show-all').click()

        page.locator('[data-metric="lowCarbon"]').click()
        page.locator('[data-panel="mix"]').click()
        page.locator('[data-chart="power"]').click()
        for width in [390,768,1024]:
            page.set_viewport_size({'width':width,'height':844})
            page.wait_for_function('Math.abs(document.querySelector("#history-chart").viewBox.baseVal.width - document.querySelector("#chart-wrap").clientWidth) < 1')
            assert not page.evaluate('document.documentElement.scrollWidth > innerWidth'), f'Overflow at {width}px'
            page.screenshot(path=str(OUT/f'viewport-{width}.png'),full_page=True)
        assert not errors, errors
        assert not external, f'Unexpected runtime external requests: {external}'
        print(json.dumps({'browser_errors':errors,'external_requests':external,'viewports':[390,768,1024,1440],
                          'journeys':['country','metric','flows','fractional zoom','pan','timeline','playback','URL state','price zones','charts','download','methodology','rankings']}))
        browser.close()


if __name__ == '__main__':
    check()
