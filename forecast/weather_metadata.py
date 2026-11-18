"""Validate the six model inputs before assigning grid points to DK1 and DK2."""

from math import isfinite

from .features import CONFIG

UNITS = {"temperature_2m": "°C", "wind_speed_100m": "m/s", "wind_direction_100m": "°",
         "shortwave_radiation": "W/m²", "cloud_cover": "%", "terrestrial_radiation": "W/m²"}


def validate_locations(points):
    if not isinstance(points, list) or len(points) != len(CONFIG["locations"]):
        raise ValueError("Unexpected weather locations")
    for point, location in zip(points, CONFIG["locations"], strict=True):
        if not isinstance(point, dict):
            raise ValueError("Invalid weather point metadata")
        for key in ("latitude", "longitude"):
            coord = point.get(key)
            if not isinstance(coord, (int, float)) or not isfinite(coord) or abs(coord - location[key]) > .4:
                raise ValueError("Weather location does not match the requested grid point")
        units = point.get("hourly_units") or {}
        if point.get("utc_offset_seconds") != 0 or any(units.get(v) != unit for v, unit in UNITS.items()):
            raise ValueError("Unexpected weather units or UTC offset")
