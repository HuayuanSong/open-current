"""Fixture-backed checks for country transitions, refresh recovery and chart state."""

import copy
import datetime as dt
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import threading
import unittest
from urllib.parse import urlparse

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
URL = os.environ.get("GRID_TEST_URL")


def fixture(name):
    return json.loads((ROOT / "dist/data" / name).read_text())


def transition_fixture():
    data = fixture("forecasts.json")
    record = data["latest"]
    origin = dt.datetime(2026, 11, 1, 23, tzinfo=dt.timezone.utc)
    record["origin"] = origin.isoformat()
    record["issuedAt"] = (origin + dt.timedelta(minutes=20)).isoformat()
    record["times"] = [
        (origin + dt.timedelta(hours=hour)).isoformat() for hour in range(1, 49)
    ]
    record["ranges"]["issuedAt"] = record["issuedAt"]
    data["comparison"] = None
    data["observations"] = {
        area: [
            {
                "time": (origin - dt.timedelta(hours=offset)).isoformat(),
                "area": area,
                "wind": 250,
                "solar": 0,
                "load": 2000,
            }
            for offset in [1, 0]
        ]
        for area in ["DK1", "DK2"]
    }
    return data


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, message, *args):
        pass


class BrowserRegressions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.url = URL
        cls.server = None
        if cls.url is None:
            cls.server = ThreadingHTTPServer(
                ("127.0.0.1", 0), partial(QuietHandler, directory=str(ROOT / "dist"))
            )
            cls.server_thread = threading.Thread(
                target=cls.server.serve_forever, daemon=True
            )
            cls.server_thread.start()
            cls.url = f"http://127.0.0.1:{cls.server.server_port}/"
        cls.playwright = sync_playwright().start()
        cls.browser = cls.playwright.chromium.launch(
            executable_path=os.environ.get("PLAYWRIGHT_EXECUTABLE_PATH")
        )

    @classmethod
    def tearDownClass(cls):
        cls.browser.close()
        cls.playwright.stop()
        if cls.server:
            cls.server.shutdown()
            cls.server.server_close()
            cls.server_thread.join(timeout=5)

    def setUp(self):
        self.page = self.browser.new_page(viewport={"width": 1440, "height": 1050})
        self.errors = []
        self.pending_routes = {}
        self.external = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.on(
            "request",
            lambda request: (
                self.external.append(request.url)
                if request.url.startswith("http")
                and urlparse(request.url).netloc != urlparse(self.url).netloc
                else None
            ),
        )

    def tearDown(self):
        for route in self.pending_routes.values():
            route.abort()
        self.page.close()
        self.assertEqual(self.errors, [])
        self.assertEqual(self.external, [])

    def load(self):
        self.page.goto(self.url, wait_until="networkidle")
        expect(self.page.locator("#country-code")).to_have_text("DK")
        expect(self.page.locator("#download-data")).to_be_enabled()

    def test_missing_generation_preserves_independent_feeds_and_forecasts(self):
        country = fixture("countries/dk.json")
        country["power"] = []
        index = fixture("index.json")
        index["countries"]["dk"]["power"] = None
        self.page.route(
            "**/data/countries/dk.json", lambda route: route.fulfill(json=country)
        )
        self.page.route("**/data/index.json", lambda route: route.fulfill(json=index))
        self.load()
        expect(self.page.locator("#detail-value")).to_have_text("—")
        expect(self.page.locator("#country-time")).to_have_text("No reading in this period")
        self.page.locator('[data-chart="power"]').click()
        expect(self.page.locator("#chart-empty")).to_be_visible()
        expect(self.page.locator(".chart-line")).to_have_count(0)
        expect(self.page.locator(".danish-card")).to_have_count(2)
        for card in self.page.locator(".danish-card").all():
            expect(card.locator(".co2-value")).not_to_contain_text("—")
            expect(card.locator("time")).to_contain_text("UTC")
        expect(self.page.locator(".forecast-predicted")).to_have_count(1)
        expect(self.page.locator("#forecast-download")).to_be_enabled()
        self.page.locator('[data-panel="flows"]').click()
        self.assertGreater(self.page.locator("#flows-list .flow-row").count(), 0)
        self.page.locator('[data-metric="net"]').click()
        expect(self.page.locator("#detail-value")).to_contain_text("GW")
        self.page.locator('[data-chart="price"]').click()
        expect(self.page.locator(".chart-line")).to_have_count(1)
        self.assert_download_country("dk")

    def assert_download_country(self, country):
        with self.page.expect_download() as captured:
            self.page.locator("#download-data").click()
        download = captured.value
        self.assertEqual(download.suggested_filename, f"open-current-{country}.json")
        self.assertEqual(json.loads(Path(download.path()).read_text())["code"], country)

    def test_country_loading_and_failure_hide_previous_country_data(self):
        self.page.route(
            "**/data/countries/fr.json",
            lambda route: self.pending_routes.setdefault("fr", route),
        )
        self.load()
        self.page.locator("#country-select").select_option("fr")
        expect(self.page.locator("#country-time")).to_have_text("Loading readings…")
        expect(self.page.locator("#download-data")).to_be_disabled()
        expect(self.page.locator(".chart-line")).to_have_count(0)
        expect(self.page.locator("#price-current")).to_have_text("—")
        expect(self.page.locator("#price-zone")).to_be_disabled()
        self.pending_routes.pop("fr").fulfill(
            status=503, body="temporary fixture failure"
        )
        expect(self.page.locator("#country-time")).to_have_text(
            "Could not load this country"
        )
        expect(self.page.locator(".chart-line")).to_have_count(0)
        expect(self.page.locator("#download-data")).to_be_disabled()
        expect(self.page.locator("#price-zone")).not_to_contain_text("Denmark")
        self.page.locator("#country-select").select_option("de")
        expect(self.page.locator("#download-data")).to_be_enabled()
        self.assert_download_country("de")

    def test_late_country_response_cannot_enable_a_different_country_download(self):
        self.page.route(
            "**/data/countries/fr.json",
            lambda route: self.pending_routes.setdefault("fr", route),
        )
        self.page.route(
            "**/data/countries/de.json",
            lambda route: self.pending_routes.setdefault("de", route),
        )
        self.load()
        self.page.locator("#country-select").select_option("fr")
        self.page.locator("#country-select").select_option("de")
        self.pending_routes.pop("fr").fulfill(json=fixture("countries/fr.json"))
        expect(self.page.locator("#country-code")).to_have_text("DE")
        expect(self.page.locator("#download-data")).to_be_disabled()
        expect(self.page.locator(".chart-line")).to_have_count(0)
        self.pending_routes.pop("de").fulfill(json=fixture("countries/de.json"))
        expect(self.page.locator("#download-data")).to_be_enabled()
        self.assert_download_country("de")

    def test_mismatched_country_response_is_rejected(self):
        self.page.route(
            "**/data/countries/fr.json",
            lambda route: route.fulfill(json=fixture("countries/dk.json")),
        )
        self.load()
        self.page.locator("#country-select").select_option("fr")
        expect(self.page.locator("#country-time")).to_have_text(
            "Could not load this country"
        )
        expect(self.page.locator("#download-data")).to_be_disabled()
        expect(self.page.locator(".chart-line")).to_have_count(0)
        expect(self.page.locator("#price-current")).to_have_text("—")

    def test_failed_country_load_recovers_without_a_new_index(self):
        self.page.clock.install(
            time=dt.datetime(2026, 11, 1, 12, tzinfo=dt.timezone.utc)
        )
        calls = 0

        def country_response(route):
            nonlocal calls
            calls += 1
            if calls == 1:
                route.fulfill(status=503, body="temporary fixture failure")
            else:
                route.fulfill(json=fixture("countries/dk.json"))

        self.page.route("**/data/countries/dk.json", country_response)
        self.page.goto(self.url, wait_until="networkidle")
        expect(self.page.locator("#download-data")).to_be_disabled()
        original_label = self.page.locator("#collection-time").inner_text()
        self.page.clock.fast_forward(300001)
        expect(self.page.locator("#download-data")).to_be_enabled()
        expect(self.page.locator("#collection-time")).to_have_text(original_label)
        self.assertEqual(calls, 2)
        self.assert_download_country("dk")

    def test_incomplete_snapshot_refresh_retries_the_same_timestamp(self):
        self.page.clock.install(
            time=dt.datetime(2026, 11, 1, 12, tzinfo=dt.timezone.utc)
        )
        original = fixture("index.json")
        updated = copy.deepcopy(original)
        updated["generatedAt"] = (
            dt.datetime.fromisoformat(original["generatedAt"].replace("Z", "+00:00"))
            + dt.timedelta(hours=1)
        ).isoformat()
        calls = {"index": 0, "denmark": 0, "country": 0}

        def index_response(route):
            calls["index"] += 1
            route.fulfill(json=original if calls["index"] == 1 else updated)

        def denmark_response(route):
            calls["denmark"] += 1
            if calls["denmark"] == 2:
                route.fulfill(status=503, body="temporary fixture failure")
            else:
                route.fulfill(json=fixture("denmark.json"))

        def country_response(route):
            calls["country"] += 1
            route.fulfill(json=fixture("countries/dk.json"))

        self.page.route("**/data/index.json", index_response)
        self.page.route("**/data/denmark.json", denmark_response)
        self.page.route("**/data/countries/dk.json", country_response)
        self.load()
        original_label = self.page.locator("#collection-time").inner_text()
        with self.page.expect_response(
            lambda response: response.url.endswith("/data/denmark.json")
            and response.status == 503
        ):
            self.page.clock.fast_forward(300001)
        expect(self.page.locator("#collection-time")).to_have_text(original_label)
        with self.page.expect_response(
            lambda response: response.url.endswith("/data/index.json")
        ):
            self.page.clock.fast_forward(300001)
        expect(self.page.locator("#collection-time")).not_to_have_text(original_label)
        expect(self.page.locator("#download-data")).to_be_enabled()
        self.assertEqual(calls, {"index": 3, "denmark": 3, "country": 2})

    def test_price_expires_locally_while_historical_selection_is_preserved(self):
        now = dt.datetime(2026, 11, 1, 12, 14, tzinfo=dt.timezone.utc)
        self.page.clock.install(time=now)
        index = fixture("index.json")
        index["generatedAt"] = now.isoformat()
        country = fixture("countries/dk.json")
        stamp = int(now.replace(minute=0).timestamp() * 1000)
        country["prices"]["DK1"]["rows"] = [{"t": stamp, "value": 17.25}]
        country["prices"]["DK1"]["intervalMinutes"] = 15
        calls = {"index": 0, "country": 0}

        def index_response(route):
            calls["index"] += 1
            route.fulfill(json=index)

        def country_response(route):
            calls["country"] += 1
            route.fulfill(json=country)

        self.page.route("**/data/index.json", index_response)
        self.page.route("**/data/countries/dk.json", country_response)
        self.load()
        expect(self.page.locator("#price-current")).to_contain_text("17.25")
        self.page.clock.fast_forward(61000)
        expect(self.page.locator("#price-current")).to_have_text("—")
        self.assertEqual(calls, {"index": 1, "country": 1})
        self.page.locator("#time-slider").evaluate(
            "el => { el.value = 167; el.dispatchEvent(new Event('input', {bubbles:true})); }"
        )
        expect(self.page.locator("#price-current")).to_contain_text("17.25")
        self.page.clock.fast_forward(61000)
        expect(self.page.locator("#price-current")).to_contain_text("17.25")
        self.assertEqual(calls, {"index": 1, "country": 1})

    def test_price_timer_handles_zero_negative_values_and_a_gap(self):
        now = dt.datetime(2026, 11, 1, 12, 0, 50, tzinfo=dt.timezone.utc)
        self.page.clock.install(time=now)
        country = fixture("countries/dk.json")
        stamp = int(now.replace(second=0).timestamp() * 1000)
        country["prices"]["DK1"].update(
            intervalMinutes=1,
            rows=[
                {"t": stamp, "value": 0},
                {"t": stamp + 60000, "value": -12.5},
                {"t": stamp + 180000, "value": 25.75},
            ],
        )
        calls = 0

        def country_response(route):
            nonlocal calls
            calls += 1
            route.fulfill(json=country)

        self.page.route("**/data/countries/dk.json", country_response)
        self.load()
        expect(self.page.locator("#price-current")).to_contain_text("0.00")
        self.page.clock.fast_forward(11000)
        expect(self.page.locator("#price-current")).to_contain_text("-12.50")
        self.page.clock.fast_forward(60000)
        expect(self.page.locator("#price-current")).to_have_text("—")
        self.page.clock.fast_forward(60000)
        expect(self.page.locator("#price-current")).to_contain_text("25.75")
        self.page.clock.fast_forward(60000)
        expect(self.page.locator("#price-current")).to_have_text("—")
        self.assertEqual(calls, 1)

    def test_empty_forecast_comparison_cannot_reveal_previous_outlook_tooltips(self):
        data = fixture("forecasts.json")
        data["comparison"] = None
        self.page.route(
            "**/data/forecasts.json", lambda route: route.fulfill(json=data)
        )
        self.load()
        expect(self.page.locator(".forecast-predicted")).to_have_count(1)
        self.page.locator('[data-forecast-view="comparison"]').click()
        expect(self.page.locator("#forecast-empty")).to_be_visible()
        expect(self.page.locator(".forecast-predicted")).to_have_count(0)
        chart = self.page.locator("#forecast-chart")
        chart.focus()
        self.page.keyboard.press("ArrowRight")
        expect(self.page.locator("#forecast-tooltip")).to_be_hidden()
        chart.dispatch_event("pointermove", {"clientX": 500, "clientY": 100})
        expect(self.page.locator("#forecast-tooltip")).to_be_hidden()
        self.page.locator('[data-forecast-view="outlook"]').click()
        chart.focus()
        self.page.keyboard.press("ArrowRight")
        expect(self.page.locator("#forecast-tooltip")).to_be_visible()

    def test_forecast_transition_labels_hour_intervals_and_preserves_download(self):
        data = transition_fixture()
        self.page.route(
            "**/data/forecasts.json", lambda route: route.fulfill(json=data)
        )
        self.load()
        expect(self.page.locator(".forecast-transition")).to_have_count(1)
        expect(self.page.locator("#forecast-transition-key")).to_be_visible()
        expect(self.page.locator(".forecast-transition title")).to_contain_text(
            "Visual transition"
        )
        chart = self.page.locator("#forecast-chart")
        tooltip = self.page.locator("#forecast-tooltip")
        chart.focus()
        self.page.keyboard.press("ArrowRight")
        expect(tooltip).to_contain_text("22:00")
        expect(tooltip).to_contain_text("23:00")
        expect(tooltip).to_contain_text("hourly average")
        expect(tooltip).to_contain_text("Observed: 250 MW")
        expect(tooltip).to_contain_text("Forecast: —")
        expect(tooltip).to_contain_text("visual transition only")
        self.page.keyboard.press("ArrowRight")
        expect(tooltip).to_contain_text("00:00")
        label = "".join(tooltip.locator("div").first.inner_text().split())
        self.assertIn("1Nov", label)
        self.assertIn("2Nov", label)
        with self.page.expect_download() as captured:
            self.page.locator("#forecast-download").click()
        self.assertEqual(
            json.loads(Path(captured.value.path()).read_text()), data["latest"]
        )
        for width in [320, 390, 1440]:
            self.page.set_viewport_size({"width": width, "height": 1050})
            self.page.wait_for_function(
                'Math.abs(document.querySelector("#forecast-chart").viewBox.baseVal.width - document.querySelector("#forecast-chart-wrap").clientWidth) < 1'
            )
            chart.focus()
            self.page.keyboard.press("ArrowLeft")
            self.page.keyboard.press("ArrowRight")
            expect(tooltip).to_be_visible()
            box = tooltip.bounding_box()
            self.assertGreaterEqual(box["x"], 0)
            self.assertLessEqual(box["x"] + box["width"], width)
            self.assertFalse(
                self.page.evaluate("document.documentElement.scrollWidth > innerWidth")
            )
        self.page.locator("#forecast-range-toggle").uncheck()
        expect(self.page.locator(".forecast-transition")).to_have_count(1)
        self.page.locator('[data-forecast-view="comparison"]').click()
        expect(self.page.locator(".forecast-transition")).to_have_count(0)
        expect(self.page.locator("#forecast-transition-key")).to_be_hidden()

    def test_forecast_transition_does_not_bridge_gaps_or_overlapping_readings(self):
        data = transition_fixture()
        self.page.route(
            "**/data/forecasts.json", lambda route: route.fulfill(json=data)
        )
        for case in ["missing", "unknown", "missing prediction", "overlap"]:
            with self.subTest(case=case):
                data = transition_fixture()
                rows = data["observations"]["DK1"]
                if case == "missing":
                    rows.pop()
                elif case == "unknown":
                    rows[-1]["wind"] = None
                elif case == "missing prediction":
                    data["latest"]["areas"]["DK1"]["wind"][0] = None
                else:
                    rows.append(
                        {**rows[-1], "time": data["latest"]["times"][0], "wind": 0}
                    )
                self.page.goto(self.url, wait_until="networkidle")
                self.page.locator('[data-forecast-area="DK1"]').click()
                expect(self.page.locator(".forecast-predicted")).to_have_count(1)
                expect(self.page.locator(".forecast-transition")).to_have_count(0)
                expect(self.page.locator("#forecast-transition-key")).to_be_hidden()

    def test_forecast_still_renders_with_cached_html_without_transition_legend(self):
        data = transition_fixture()
        self.page.route(
            "**/data/forecasts.json", lambda route: route.fulfill(json=data)
        )
        # An older cached page can load newer script contents during deployment.
        self.page.add_init_script(
            "document.addEventListener('readystatechange', () => {"
            "if (document.readyState === 'interactive') "
            "document.querySelector('#forecast-transition-key')?.remove();"
            "});"
        )
        self.load()
        expect(self.page.locator("#forecast-transition-key")).to_have_count(0)
        expect(self.page.locator(".forecast-predicted")).to_have_count(1)
        expect(self.page.locator(".forecast-range-band")).to_have_count(1)
        expect(self.page.locator("#forecast-download")).to_be_enabled()


if __name__ == "__main__":
    unittest.main(verbosity=2)
