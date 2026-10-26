"""Explicit coverage and public-redistribution allowlist."""
COUNTRIES = {
    'dk': ('Denmark', 10.0, 56.1), 'de': ('Germany', 10.4, 51.1),
    'se': ('Sweden', 15.0, 62.0), 'no': ('Norway', 8.4, 62.0),
    'fi': ('Finland', 26.0, 64.0), 'fr': ('France', 2.3, 46.7),
    'nl': ('Netherlands', 5.4, 52.2), 'be': ('Belgium', 4.7, 50.8),
    'at': ('Austria', 14.0, 47.6), 'ch': ('Switzerland', 8.2, 46.8),
    'pl': ('Poland', 19.0, 52.0), 'cz': ('Czechia', 15.4, 49.8),
    'it': ('Italy', 12.8, 42.7), 'es': ('Spain', -3.5, 40.1),
    'pt': ('Portugal', -8.0, 39.6), 'ie': ('Ireland', -8.0, 53.4),
    'uk': ('United Kingdom', -2.5, 54.3), 'ee': ('Estonia', 25.5, 58.7),
    'lv': ('Latvia', 24.8, 56.9), 'lt': ('Lithuania', 23.9, 55.2),
    'lu': ('Luxembourg', 6.1, 49.8), 'hu': ('Hungary', 19.3, 47.2),
    'sk': ('Slovakia', 19.5, 48.7), 'si': ('Slovenia', 14.9, 46.1),
    'hr': ('Croatia', 16.5, 45.2), 'ro': ('Romania', 24.8, 45.9),
    'bg': ('Bulgaria', 25.3, 42.7), 'gr': ('Greece', 23.0, 39.0),
    'cy': ('Cyprus', 33.1, 35.0), 'ba': ('Bosnia and Herzegovina', 17.8, 44.0),
    'rs': ('Serbia', 20.7, 44.0), 'me': ('Montenegro', 19.3, 42.8),
    'mk': ('North Macedonia', 21.7, 41.6), 'xk': ('Kosovo', 21.0, 42.6),
}

# Energy-Charts licenses other price zones for private/internal use only.
PRICE_ZONES = {
    'DK1': ('dk', 'West Denmark'), 'DK2': ('dk', 'East Denmark'),
    'DE-LU': ('de', 'Germany / Luxembourg'), 'FR': ('fr', 'France'),
    'AT': ('at', 'Austria'), 'BE': ('be', 'Belgium'), 'CH': ('ch', 'Switzerland'),
    'CZ': ('cz', 'Czechia'), 'HU': ('hu', 'Hungary'),
    'IT-North': ('it', 'Northern Italy'), 'NL': ('nl', 'Netherlands'),
    'NO2': ('no', 'Southwest Norway'), 'PL': ('pl', 'Poland'),
    'SE4': ('se', 'Southern Sweden'), 'SI': ('si', 'Slovenia'),
}

SOURCE = 'https://api.energy-charts.info/v2/'

