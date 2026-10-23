# app.py – Flut-App 1.1 · Weltweite Meeresspiegel- und Flutungssimulation
# Badewannen-Modell auf Basis der AWS Terrain Tiles (inkl. Meeresboden).
# 1.0: Reiter Burgen (OSM). 1.1: Klick auf Burg öffnet Info-Fenster, kompakter Code.
from __future__ import annotations

import base64
import html
import io
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import folium
import matplotlib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components
from folium import plugins
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from streamlit_folium import st_folium

try:
    from skimage.measure import find_contours
    from skimage.morphology import reconstruction
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

st.set_page_config(page_title="Flut-App", page_icon="🌊", layout="wide")
RS = getattr(Image, "Resampling", Image)

# ============================== Konstanten ==============================
APP_VERSION = "1.1"
LEVEL_MIN, LEVEL_MAX = -11000.0, 5000.0
DEFAULT_CENTER, DEFAULT_ZOOM, MAP_HEIGHT = [50.0, 8.0], 5, 720
USER_AGENT = "Flut-App (privates Projekt)"
TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TOPO_URL = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"
MAX_TILE_ZOOM, TILE, MAX_TILES = 15, 256, 24
EARTH_CIRC, MAX_LAT = 40075016.686, 85.0511
OCEAN_AREA_KM2, OCEAN_VOL_KM3, ICE_SLE_M = 3.61e8, 1.335e9, 68.0

MODES = ["Wassertiefe", "Veränderung zu heute"]
FLOOD_MODES = ["Badewanne (alles unter dem Wasserstand)", "Zusammenhängend vom Kartenrand",
               "Ab Startpunkt (letzter Klick)"]
WATER_CMAP = LinearSegmentedColormap.from_list("wasser", ["#7cc8f8", "#1f78d1", "#08306b"])
COL_NEW, COL_DRY, COL_OLD = (30, 120, 220), (224, 170, 70), (150, 190, 230)
COL_SHORE, COL_L2, COL_BAND = (255, 215, 0), (220, 40, 40), (255, 235, 120)

TEXTURES_3D = {"Satellit (Esri)": ESRI_URL, "OpenStreetMap": OSM_URL, "Gelände (OpenTopoMap)": TOPO_URL}
TEX_EXACT = ["Satellit (Esri)", "OpenStreetMap", "Gelände (OpenTopoMap)", "Höhenfarben", "Relief (grau)"]
RES_3D = [150, 200, 250, 300, 350]
BASEMAPS_ANIM = {"Satellit (Esri)": (ESRI_URL, "© Esri World Imagery"),
                 "OpenStreetMap": (OSM_URL, "© OpenStreetMap-Mitwirkende"),
                 "Gelände (OpenTopoMap)": (TOPO_URL, "© OpenTopoMap, © OSM"), "Nur Relief": (None, "")}
BASE_MAXZ = {"Satellit (Esri)": 18, "OpenStreetMap": 18, "Gelände (OpenTopoMap)": 17}
ANIM_SPEEDS = {"langsam": 700, "mittel": 400, "schnell": 200}
ANIM_QUALITY = {"Standard (1000 px)": 1000, "Hoch (1400 px)": 1400, "Sehr hoch (1800 px)": 1800}

EU_COUNTRIES = {
    "Deutschland": "DE", "Österreich": "AT", "Schweiz": "CH", "Liechtenstein": "LI", "Frankreich": "FR",
    "Belgien": "BE", "Niederlande": "NL", "Luxemburg": "LU", "Italien": "IT", "San Marino": "SM",
    "Spanien": "ES", "Portugal": "PT", "Andorra": "AD", "Vereinigtes Königreich": "GB", "Irland": "IE",
    "Dänemark": "DK", "Norwegen": "NO", "Schweden": "SE", "Finnland": "FI", "Island": "IS", "Polen": "PL",
    "Tschechien": "CZ", "Slowakei": "SK", "Ungarn": "HU", "Slowenien": "SI", "Kroatien": "HR",
    "Bosnien und Herzegowina": "BA", "Serbien": "RS", "Montenegro": "ME", "Albanien": "AL",
    "Nordmazedonien": "MK", "Griechenland": "GR", "Bulgarien": "BG", "Rumänien": "RO", "Moldau": "MD",
    "Ukraine": "UA", "Belarus": "BY", "Litauen": "LT", "Lettland": "LV", "Estland": "EE", "Malta": "MT",
    "Zypern": "CY", "Monaco": "MC"}
CC_NAME = {v: k for k, v in EU_COUNTRIES.items()}
CC_NAME["XX"] = "eigene Liste"
CASTLE_EXCLUDE = {"palace", "stately", "manor", "schloss", "fortress"}
OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"]
CASTLE_FILE = "burgen_europa.geojson"
CASTLE_COLS = ["id", "name", "typ", "ruine", "land", "lat", "lon", "wikipedia", "wikidata", "baujahr"]
CASTLE_COLORS = {"unter Wasser": "#d62828", "am Ufer": "#1f78d1", "trocken": "#8a8a8a", "unbekannt": "#8b5a2b"}
CASTLE_AREAS = ["Alle geladenen Burgen", "Nur sichtbarer Kartenausschnitt"]
CASTLE_Z = {10: "grob (~100 m, schnell)", 11: "mittel (~50 m)", 12: "fein (~25 m, langsam)"}
CASTLE_CB = """function (row) {
  var icon = L.divIcon({className: '', iconSize: [14, 14], html: '<div style="width:10px;height:10px;border-radius:50%;background:__C__;border:2px solid #fff;box-shadow:0 0 2px rgba(0,0,0,.7)"></div>'});
  var m = L.marker(new L.LatLng(row[0], row[1]), {icon: icon});
  m.bindTooltip(row[2]);
  m.bindPopup(row[3], {maxWidth: 300});
  return m;
}"""
CASTLE_ICON = """function (cluster) {
  var n = cluster.getChildCount(), s = n < 10 ? 26 : (n < 100 ? 32 : (n < 1000 ? 38 : 46));
  return L.divIcon({className: '', iconSize: [s, s], html: '<div style="background:__C__;color:#fff;border-radius:50%;width:' + s + 'px;height:' + s + 'px;line-height:' + s + 'px;text-align:center;font-weight:600;font-size:12px;border:2px solid #fff;box-shadow:0 0 4px rgba(0,0,0,.5)">' + n + '</div>'});
}"""

PRESETS = [("🌍 Heute", 0.0), ("🧊 Eiszeit −120 m", -120.0), ("🔥 Eis geschmolzen +70 m", 70.0)]
PLACES = {
    "Doggerland, Nordsee (−120 m)": dict(center=[54.5, 3.0], zoom=6, level=-120.0),
    "Beringstraße (−120 m)": dict(center=[65.5, -169.0], zoom=5, level=-120.0),
    "Mittelmeer (−120 m)": dict(center=[38.5, 15.0], zoom=5, level=-120.0),
    "Norddeutschland / Niederlande (+10 m)": dict(center=[52.8, 7.0], zoom=7, level=10.0),
    "Mittelrheintal (Burgen)": dict(center=[50.15, 7.7], zoom=10, level=100.0),
    "Lake Bonneville, Utah (ca. 1550 m)": dict(center=[40.8, -112.8], zoom=7, level=1550.0),
    "Glacial Lake Missoula, Montana (ca. 1280 m)": dict(center=[47.1, -114.2], zoom=7, level=1280.0)}

S = st.session_state
for _k, _v in dict(
        level=0.0, level_input=0.0, level_slider=0.0, center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM,
        view=None, calc_view=None, mode=MODES[0], flood_mode=FLOOD_MODES[0], alpha=0.65,
        shore=True, auto=True, click=None, click_on=True, place=list(PLACES)[0], _rr=0,
        hill_on=False, hill_op=0.45, band_on=False, band_m=10.0, l2_on=False, level2=100.0,
        sinks_on=False, sinks_min=2.0, wms_url="", wms_layers="", xyz_url="", ovl_op=0.6,
        markers=[], drawn_line=None, areas={}, anim=None, terr=None,
        tex3e=TEX_EXACT[0], exag3e=3.0, res3e=250, walpha3e=0.8,
        anim_base=list(BASEMAPS_ANIM)[0], anim_hill=0.3, anim_lo=-100.0, anim_hi=100.0,
        anim_steps=16, anim_speed="mittel", anim_q="Hoch (1400 px)",
        castles={}, castle_sel=["Deutschland"], castle_map=True, castle_dry=True, castle_r=500,
        castle_z=11, castle_area=CASTLE_AREAS[0], castle_filter=["unter Wasser", "am Ufer", "trocken"]).items():
    if _k not in S:
        S[_k] = _v.copy() if isinstance(_v, (list, dict)) else _v
if S.res3e not in RES_3D:
    S.res3e = 250

if not S.get("_qp_done"):
    S["_qp_done"] = True
    try:
        qp = st.query_params
        if "lat" in qp and "lon" in qp:
            S.center = [float(qp["lat"]), float(qp["lon"])]
        if "zoom" in qp:
            S.zoom = int(float(qp["zoom"]))
        if "level" in qp:
            _lv = float(np.clip(float(qp["level"]), LEVEL_MIN, LEVEL_MAX))
            S.level = S.level_input = S.level_slider = _lv
    except Exception:
        pass

st.markdown("""<style>
.block-container {padding-top: 3.2rem; max-width: 1800px;}
.app-header {background: linear-gradient(135deg, #0b3d6b 0%, #1f6fb2 55%, #5fb0e5 100%);
  color: #fff; padding: 16px 24px; border-radius: 14px; margin-bottom: 14px; box-shadow: 0 4px 14px rgba(11,61,107,.25);}
.app-header h1 {color: #fff !important; margin: 0 !important; padding: 2px 0 !important;
  font-size: 1.8rem; line-height: 1.35 !important; overflow: visible;}
.app-header p {margin: 4px 0 0; opacity: .92;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #d6e4ef; border-radius: 12px;
  padding: 8px 14px; box-shadow: 0 1px 4px rgba(0,0,0,.05);}
.stTabs [data-baseweb="tab-list"] {gap: 6px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {background: #e6f0f7; border-radius: 10px 10px 0 0; padding: 8px 16px;}
.stTabs [aria-selected="true"] {background: #1f6fb2 !important;}
.stTabs [aria-selected="true"] p {color: #fff !important; font-weight: 600;}
.legend span {display: inline-block; width: 14px; height: 14px; border-radius: 3px; margin: 0 6px -2px 14px; border: 1px solid #999;}
footer {visibility: hidden;}
</style>""", unsafe_allow_html=True)

# ============================== Höhendaten ==============================
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def lonlat_to_tile(lon, lat, z):
    n = 2.0 ** z
    lat = np.clip(np.asarray(lat, float), -MAX_LAT, MAX_LAT)
    return (np.asarray(lon, float) + 180.0) / 360.0 * n, (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n


def tile_to_lonlat(x, y, z):
    n = 2.0 ** z
    lon = np.asarray(x, float) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * np.asarray(y, float) / n))))
    return lon, lat


def _decode_terrarium(content):
    a = np.asarray(Image.open(io.BytesIO(content)).convert("RGB"), dtype=np.float32)
    return (a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=800)
def _fetch_tile(z, x, y):
    r = _session.get(TERRARIUM_URL.format(z=z, x=x, y=y), timeout=20)
    r.raise_for_status()
    return _decode_terrarium(r.content)


def get_tile(z, x, y):
    n = 2 ** z
    if not 0 <= y < n:
        return None
    try:
        return _fetch_tile(z, x % n, y)
    except Exception:
        return None


def normalize_bounds(south, west, north, east):
    south, north = max(south, -MAX_LAT), min(north, MAX_LAT)
    if east - west >= 360:
        return south, -180.0, north, 180.0
    shift = math.floor((west + east) / 2 / 360 + 0.5) * 360
    return south, max(west - shift, -180.0), north, min(east - shift, 180.0)


def choose_tiles(bounds, view_zoom, max_tiles=MAX_TILES):
    s, w, n, e = bounds
    for z in range(int(min(MAX_TILE_ZOOM, max(view_zoom, 0))), -1, -1):
        x0, y0 = lonlat_to_tile(w, n, z)
        x1, y1 = lonlat_to_tile(e, s, z)
        lim = 2 ** z - 1
        tx0, ty0 = max(int(math.floor(float(x0))), 0), max(int(math.floor(float(y0))), 0)
        tx1, ty1 = min(int(math.floor(float(x1))), lim), min(int(math.floor(float(y1))), lim)
        if (tx1 - tx0 + 1) * (ty1 - ty0 + 1) <= max_tiles:
            return (z, tx0, ty0, tx1, ty1)
    return (0, 0, 0, 0, 0)


@st.cache_data(show_spinner="Lade Höhendaten …", max_entries=12)
def load_mosaic(z, tx0, ty0, tx1, ty1):
    rows, missing = [], 0
    for ty in range(ty0, ty1 + 1):
        row = []
        for tx in range(tx0, tx1 + 1):
            t = get_tile(z, tx, ty)
            if t is None:
                t = np.full((TILE, TILE), np.nan, np.float32)
                missing += 1
            row.append(t)
        rows.append(np.hstack(row))
    lon_w, lat_n = tile_to_lonlat(tx0, ty0, z)
    lon_e, lat_s = tile_to_lonlat(tx1 + 1, ty1 + 1, z)
    return np.vstack(rows), (float(lat_s), float(lon_w), float(lat_n), float(lon_e)), missing


def pixel_size_m(z):
    return EARTH_CIRC / (TILE * 2 ** z)


def row_px(tiles, nrows):
    y = tiles[2] + (np.arange(nrows) + 0.5) / TILE
    return pixel_size_m(tiles[0]) * np.cos(np.radians(tile_to_lonlat(0.0, y, tiles[0])[1]))


def row_area_km2(tiles, nrows):
    return row_px(tiles, nrows) ** 2 / 1e6


def lonlat_to_pixel(tiles, lat, lon):
    x, y = lonlat_to_tile(lon, lat, tiles[0])
    return (float(y) - tiles[2]) * TILE, (float(x) - tiles[1]) * TILE


def pixel_to_lonlat(tiles, rows, cols):
    x = tiles[1] + (np.asarray(cols, float) + 0.5) / TILE
    y = tiles[2] + (np.asarray(rows, float) + 0.5) / TILE
    return tile_to_lonlat(x, y, tiles[0])


def elevation_at(lat, lon, z):
    lon = ((lon + 180.0) % 360.0) - 180.0
    x, y = lonlat_to_tile(lon, lat, z)
    tx, ty = int(math.floor(float(x))), int(math.floor(float(y)))
    t = get_tile(z, tx, ty)
    if t is None:
        return None
    v = float(t[min(int((float(y) - ty) * TILE), TILE - 1), min(int((float(x) - tx) * TILE), TILE - 1)])
    return v if np.isfinite(v) else None


# ============================== Flutung & Analyse ==============================
def compute_water(dem, level, flood_mode, seed_rc=None):
    valid = np.isfinite(dem)
    with np.errstate(invalid="ignore"):
        below = valid & (dem < level)
    if flood_mode == FLOOD_MODES[0] or not below.any():
        return below
    lab, _ = ndimage.label(below)
    if flood_mode == FLOOD_MODES[1]:
        ids = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    else:
        if seed_rc is None:
            return np.zeros_like(below)
        r, c = seed_rc
        if not (0 <= r < dem.shape[0] and 0 <= c < dem.shape[1]):
            return np.zeros_like(below)
        ids = np.array([lab[r, c]])
    ids = ids[ids > 0]
    return np.isin(lab, ids) if ids.size else np.zeros_like(below)


def hillshade(dem, px_rows, zf=3.0):
    z = dem.astype(np.float64)
    fill = np.nanmin(z) if np.isfinite(z).any() else 0.0
    z = np.where(np.isfinite(z), z, fill) * zf
    gy, gx = np.gradient(z)
    dzdx, dzdy = gx / px_rows[:, None], -gy / px_rows[:, None]
    az, alt = math.radians(315), math.radians(45)
    lx, ly, lz = math.sin(az) * math.cos(alt), math.cos(az) * math.cos(alt), math.sin(alt)
    return np.clip((-dzdx * lx - dzdy * ly + lz) / np.sqrt(dzdx ** 2 + dzdy ** 2 + 1), 0, 1)


def edge_of(mask):
    return ndimage.binary_dilation(mask & ~ndimage.binary_erosion(mask, border_value=1))


def water_rgba(dem, water, level, alpha):
    a = int(round(255 * alpha))
    d = np.where(water, level - np.nan_to_num(dem), 0.0)
    dref = float(np.percentile(d[water], 95)) if water.any() else 1.0
    rgba = WATER_CMAP(np.clip(d / max(dref, 1e-3), 0, 1), bytes=True)
    rgba[..., 3] = np.where(water, a, 0).astype(np.uint8)
    return rgba


def colorize(dem, water, level, mode, alpha, shore):
    valid = np.isfinite(dem)
    a = int(round(255 * alpha))
    if mode == MODES[0]:
        rgba = water_rgba(dem, water, level, alpha)
    else:
        with np.errstate(invalid="ignore"):
            sea, below = valid & (dem < 0), valid & (dem < level)
        rgba = np.zeros(dem.shape + (4,), np.uint8)
        rgba[water & sea] = (*COL_OLD, int(a * 0.45))
        rgba[water & ~sea] = (*COL_NEW, a)
        rgba[sea & ~below] = (*COL_DRY, a)
    if shore and water.any() and not water.all():
        rgba[edge_of(water)] = (*COL_SHORE, 255)
    return rgba


def to_png_bytes(rgba):
    buf = io.BytesIO()
    Image.fromarray(rgba).save(buf, format="PNG")
    return buf.getvalue()


def to_uri(png):
    return "data:image/png;base64," + base64.b64encode(png).decode()


@st.cache_data(show_spinner="Berechne Wasser …", max_entries=24)
def water_layer(tiles, level, mode, flood_mode, seed_rc, alpha, shore, level2, band, hill):
    dem, bounds, missing = load_mosaic(*tiles)
    valid = np.isfinite(dem)
    water = compute_water(dem, level, flood_mode, seed_rc)
    with np.errstate(invalid="ignore"):
        below, sea_today = valid & (dem < level), valid & (dem < 0)
    new_flood, new_dry = water & ~sea_today, sea_today & ~below
    rgba = colorize(dem, water, level, mode, alpha, False)
    a = int(round(255 * alpha))
    if band:
        with np.errstate(invalid="ignore"):
            rgba[valid & ~water & (np.abs(dem - level) <= band)] = (*COL_BAND, int(a * 0.7))
    if shore and water.any() and not water.all():
        rgba[edge_of(water)] = (*COL_SHORE, 255)
    if level2 is not None:
        with np.errstate(invalid="ignore"):
            w2 = valid & (dem < level2)
        if w2.any() and not w2.all():
            rgba[edge_of(w2)] = (*COL_L2, 255)
    ra = row_area_km2(tiles, dem.shape[0])[:, None]
    area = lambda m: float((m * ra).sum())
    depth = np.where(water, level - np.nan_to_num(dem), 0.0)
    n_bodies = int(ndimage.label(water)[1]) if water.any() else 0
    lab, n_land = ndimage.label(valid & ~water)
    n_isl, isl_area = 0, 0.0
    if n_land:
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        isl = np.setdiff1d(np.arange(1, n_land + 1), border)
        n_isl = int(isl.size)
        if n_isl:
            isl_area = float(ndimage.sum(np.broadcast_to(ra, dem.shape), lab, isl).sum())
    stats = dict(water=area(water), new_flood=area(new_flood), new_dry=area(new_dry),
                 volume=float((depth * ra * 1e6).sum() / 1e9), bodies=n_bodies, islands=n_isl, isl_area=isl_area)
    hill_png = None
    if hill:
        g = (hillshade(dem, row_px(tiles, dem.shape[0])) * 255).astype(np.uint8)
        hill_png = to_png_bytes(np.dstack([g, g, g, np.where(valid, 255, 0).astype(np.uint8)]))
    return dict(png=to_png_bytes(rgba), hill_png=hill_png, bounds=bounds, stats=stats, missing=missing)


@st.cache_data(show_spinner="Berechne abflusslose Senken …", max_entries=4)
def filled_dem(tiles):
    dem, _, _ = load_mosaic(*tiles)
    z = dem.astype(np.float64)
    top = float(np.nanmax(z)) if np.isfinite(z).any() else 0.0
    z = np.where(np.isfinite(z), z, top)
    seed = z.copy()
    seed[1:-1, 1:-1] = z.max()
    return reconstruction(seed, z, method="erosion").astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=6)
def sinks_layer(tiles, min_depth):
    dem, _, _ = load_mosaic(*tiles)
    depth = filled_dem(tiles) - dem
    with np.errstate(invalid="ignore"):
        m = np.isfinite(dem) & (depth > min_depth)
    rgba = np.zeros(dem.shape + (4,), np.uint8)
    if m.any():
        t = np.clip(depth / max(float(np.percentile(depth[m], 95)), 1e-3), 0, 1)
        rgba[m, 0], rgba[m, 1], rgba[m, 2] = 150, 60, 200
        rgba[m, 3] = (90 + 140 * t[m]).astype(np.uint8)
    return to_png_bytes(rgba)


@st.cache_data(show_spinner="Berechne Flutkurve …", max_entries=8)
def flood_curve(tiles, n=200):
    dem, _, _ = load_mosaic(*tiles)
    ra = np.broadcast_to(row_area_km2(tiles, dem.shape[0])[:, None], dem.shape)
    valid = np.isfinite(dem)
    z, w = dem[valid].astype(np.float64), ra[valid].astype(np.float64)
    if z.size == 0:
        return pd.DataFrame(columns=["wasserstand_m", "flaeche_km2"])
    o = np.argsort(z)
    z, cum = z[o], np.cumsum(w[o])
    levels = np.linspace(z[0], z[-1], n)
    idx = np.searchsorted(z, levels, side="left")
    return pd.DataFrame({"wasserstand_m": levels, "flaeche_km2": np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)})


@st.cache_data(show_spinner="Suche gehäufte Geländekanten …", max_entries=8)
def terrace_signal(tiles, bin_m):
    dem, _, _ = load_mosaic(*tiles)
    valid = np.isfinite(dem)
    if valid.sum() < 1000:
        return pd.DataFrame(columns=["hoehe_m", "kantendichte", "ebenen"])
    z = np.where(valid, dem, np.nanmin(dem)).astype(np.float64)
    px = row_px(tiles, dem.shape[0])[:, None]
    zs = ndimage.gaussian_filter(z, 1.0)
    gy, gx = np.gradient(zs)
    slope = np.degrees(np.arctan(np.hypot(gx / px, gy / px)))
    curv = np.abs(ndimage.laplace(zs)) / px ** 2
    edges = (curv > np.percentile(curv[valid], 95)) & (slope > 1.0) & valid
    flat = (slope < 2.0) & valid
    zv = z[valid]
    zmin, zmax = float(zv.min()), float(zv.max())
    bin_m = max(bin_m, (zmax - zmin) / 2000, 0.1)
    bins = np.arange(math.floor(zmin / bin_m) * bin_m, zmax + bin_m, bin_m)
    if len(bins) < 3:
        bins = np.linspace(zmin, zmax + 1e-6, 3)
    n_all = np.histogram(zv, bins)[0].astype(float)
    n_e, n_f = np.histogram(z[edges], bins)[0], np.histogram(z[flat], bins)[0]
    ok = n_all >= max(30, 0.0005 * zv.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        re_ = (n_e / n_all) / max(edges.sum() / valid.sum(), 1e-9)
        rf_ = (n_f / n_all) / max(flat.sum() / valid.sum(), 1e-9)
    re_[~ok], rf_[~ok] = np.nan, np.nan
    return pd.DataFrame({"hoehe_m": (bins[:-1] + bins[1:]) / 2, "kantendichte": re_, "ebenen": rf_})


def find_peaks(e, thr=1.3, k=6):
    peaks = [i for i in range(1, len(e) - 1)
             if np.isfinite(e[i]) and e[i] >= thr and e[i] >= np.nanmax(e[max(0, i - 2):i + 3])]
    return sorted(peaks, key=lambda i: -e[i])[:k]


@st.cache_data(show_spinner="Erzeuge Uferlinie …", max_entries=6)
def shoreline_geojson(tiles, level, flood_mode, seed_rc):
    if not SKIMAGE_OK:
        return None
    dem, _, _ = load_mosaic(*tiles)
    water = compute_water(dem, level, flood_mode, seed_rc)
    if not water.any() or water.all():
        return None
    feats = []
    for c in sorted(find_contours(water.astype(np.float32), 0.5), key=len, reverse=True)[:500]:
        if len(c) < 6:
            continue
        lon, lat = pixel_to_lonlat(tiles, c[:, 0] - 0.5, c[:, 1] - 0.5)
        feats.append({"type": "Feature", "properties": {"wasserstand_m": level},
                      "geometry": {"type": "LineString", "coordinates":
                                   [[round(float(a), 6), round(float(b), 6)] for a, b in zip(lon, lat)]}})
    return json.dumps({"type": "FeatureCollection", "features": feats}).encode("utf-8")


def haversine(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371008.8 * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="Berechne Höhenprofil …", max_entries=20)
def sample_profile(coords, n=400):
    pts = np.asarray(coords, float)
    cum = np.concatenate([[0.0], np.cumsum(haversine(pts[:-1, 1], pts[:-1, 0], pts[1:, 1], pts[1:, 0]))])
    total = float(cum[-1])
    if total <= 0:
        return None
    d = np.linspace(0, total, n)
    lon, lat = np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])
    z = int(np.clip(math.floor(math.log2(EARTH_CIRC * math.cos(math.radians(float(lat.mean())))
                                         / (TILE * max(total / n, 1.0)))), 0, MAX_TILE_ZOOM))
    elev = [elevation_at(float(a), float(b), z) for a, b in zip(lat, lon)]
    return pd.DataFrame({"distanz_km": d / 1000, "lat": lat, "lon": lon, "hoehe_m": pd.Series(elev, dtype=float)}), z


# ============================== Kartenhintergrund ==============================
@st.cache_data(show_spinner=False, max_entries=1500)
def _fetch_base_tile(url, z, x, y):
    r = _session.get(url.format(z=z, x=x, y=y), timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    if img.size != (TILE, TILE):
        img = img.resize((TILE, TILE))
    return np.asarray(img, dtype=np.uint8)


def base_mosaic(url, z, tx0, ty0, tx1, ty1):
    n = 2 ** z
    rows = []
    for ty in range(ty0, ty1 + 1):
        row = []
        for tx in range(tx0, tx1 + 1):
            try:
                row.append(_fetch_base_tile(url, z, tx % n, ty))
            except Exception:
                row.append(np.full((TILE, TILE, 3), 210, np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def base_for_area(url, basemap, tiles, target_w, max_base_tiles=64):
    z, tx0, ty0, tx1, ty1 = tiles
    ntiles = (tx1 - tx0 + 1) * (ty1 - ty0 + 1)
    w0, k = (tx1 - tx0 + 1) * TILE, 0
    while w0 * 2 ** k < target_w and ntiles * 4 ** (k + 1) <= max_base_tiles and z + k + 1 <= BASE_MAXZ.get(basemap, 17):
        k += 1
    f = 2 ** k
    return base_mosaic(url, z + k, tx0 * f, ty0 * f, (tx1 + 1) * f - 1, (ty1 + 1) * f - 1)


# ============================== Animation ==============================
def _font(size):
    try:
        return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


@st.cache_data(show_spinner="Erzeuge Animation in hoher Qualität …", max_entries=3)
def make_animation(tiles, lo, hi, steps, flood_mode, seed_rc, alpha, duration_ms, basemap, hill_mix, mode, shore, width):
    dem, _, _ = load_mosaic(*tiles)
    H, W = dem.shape
    tw = int(width)
    th = max(1, int(round(H * tw / W)))
    scale = tw / W
    valid = np.isfinite(dem)
    demf = np.where(valid, dem, float(np.nanmin(dem)) if valid.any() else 0.0).astype(np.float32)
    d = np.asarray(Image.fromarray(demf, mode="F").resize((tw, th), RS.BILINEAR), dtype=np.float32)
    vmask = np.asarray(Image.fromarray(valid.astype(np.uint8) * 255).resize((tw, th), RS.NEAREST)) > 0
    d = np.where(vmask, d, np.nan)
    hs = hillshade(d, np.interp(np.linspace(0, H - 1, th), np.arange(H), row_px(tiles, H)) / scale)
    url, attribution = BASEMAPS_ANIM.get(basemap, (None, ""))
    if url:
        bm = np.asarray(Image.fromarray(base_for_area(url, basemap, tiles, tw)).resize((tw, th), RS.LANCZOS),
                        dtype=np.float32) / 255.0
        base = np.clip(bm * ((1.0 - hill_mix) + hill_mix * (0.45 + 0.8 * hs[..., None])), 0, 1)
    else:
        base = np.dstack([(hs * 190 + 50)] * 3).astype(np.float32) / 255.0
    seed = (int(seed_rc[0] * scale), int(seed_rc[1] * scale)) if seed_rc else None
    font, small = _font(max(16, tw // 34)), _font(max(11, tw // 80))
    gif_w = min(tw, 900)
    gif_h = max(1, int(round(th * gif_w / tw)))
    frames, pils = [], []
    for lv in np.linspace(lo, hi, steps):
        top = colorize(d, compute_water(d, lv, flood_mode, seed), lv, mode, alpha, shore).astype(np.float32) / 255.0
        ta = top[..., 3:4]
        img = Image.fromarray(((top[..., :3] * ta + base * (1 - ta)) * 255 + 0.5).astype(np.uint8), "RGB")
        draw = ImageDraw.Draw(img)
        txt = f"Wasserstand {lv:+.0f} m"
        bx = draw.textbbox((0, 0), txt, font=font)
        draw.rectangle([12, 12, 32 + bx[2] - bx[0], 32 + bx[3] - bx[1]], fill=(30, 30, 30))
        draw.text((22 - bx[0], 22 - bx[1]), txt, fill=(255, 255, 255), font=font)
        if attribution:
            ab = draw.textbbox((0, 0), attribution, font=small)
            x0, y0 = img.width - (ab[2] - ab[0]) - 14, img.height - (ab[3] - ab[1]) - 12
            draw.rectangle([x0 - 5, y0 - 4, img.width - 7, img.height - 6], fill=(255, 255, 255))
            draw.text((x0 - ab[0], y0 - ab[1]), attribution, fill=(60, 60, 60), font=small)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92, optimize=True)
        frames.append((float(lv), buf.getvalue()))
        pils.append(img.resize((gif_w, gif_h), RS.LANCZOS).quantize(colors=255))
    gif = io.BytesIO()
    pils[0].save(gif, format="GIF", save_all=True, append_images=pils[1:],
                 duration=[int(duration_ms)] * (len(pils) - 1) + [int(duration_ms) * 4], loop=0)
    return frames, gif.getvalue()


PLAYER_HTML = """
<div style="font-family:sans-serif">
 <img id="im" style="width:100%;height:__H__px;object-fit:contain;background:#eef3f7;border-radius:10px">
 <div style="display:flex;gap:8px;align-items:center;margin-top:8px">
  <button id="pp" style="padding:6px 14px;border-radius:8px;border:1px solid #1f6fb2;background:#1f6fb2;color:#fff;cursor:pointer">▶ Abspielen</button>
  <button id="pv" style="padding:6px 10px;border-radius:8px;border:1px solid #bbb;cursor:pointer">⏮</button>
  <button id="nx" style="padding:6px 10px;border-radius:8px;border:1px solid #bbb;cursor:pointer">⏭</button>
  <input id="sl" type="range" min="0" max="__MAX__" value="0" style="flex:1">
  <span id="lb" style="min-width:190px;font-weight:600;color:#10233A"></span>
 </div>
</div>
<script>
const imgs = __IMGS__, lbs = __LBS__, dur = __DUR__;
const im = document.getElementById("im"), sl = document.getElementById("sl"),
      lb = document.getElementById("lb"), pp = document.getElementById("pp");
imgs.forEach(s => { const p = new Image(); p.src = s; });
let i = 0, t = null;
function show(k) { i = (k + imgs.length) % imgs.length; im.src = imgs[i]; sl.value = i; lb.textContent = lbs[i]; }
function stop() { if (t) { clearInterval(t); t = null; } pp.textContent = "▶ Abspielen"; }
pp.onclick = () => { if (t) { stop(); } else { t = setInterval(() => show(i + 1), dur); pp.textContent = "⏸ Pause"; } };
document.getElementById("pv").onclick = () => { stop(); show(i - 1); };
document.getElementById("nx").onclick = () => { stop(); show(i + 1); };
sl.oninput = () => { stop(); show(parseInt(sl.value)); };
show(0);
</script>
"""


def anim_player(frames, duration_ms, height=MAP_HEIGHT):
    imgs = ["data:image/jpeg;base64," + base64.b64encode(b).decode() for _, b in frames]
    lbs = [f"Wasserstand {de_num(lv, 0, True)} m ({k + 1}/{len(frames)})" for k, (lv, _) in enumerate(frames)]
    h_ = (PLAYER_HTML.replace("__IMGS__", json.dumps(imgs)).replace("__LBS__", json.dumps(lbs))
          .replace("__DUR__", str(int(duration_ms))).replace("__MAX__", str(len(frames) - 1))
          .replace("__H__", str(int(height))))
    components.html(h_, height=int(height) + 70)


# ============================== 3D (gleiche Daten wie Karte) ==============================
@st.cache_data(show_spinner="Bereite 3D-Ansicht vor …", max_entries=4)
def scene_3d(tiles, level, flood_mode, seed_rc, texture, n_max, mode, alpha, shore):
    dem, bounds, _ = load_mosaic(*tiles)
    water = compute_water(dem, level, flood_mode, seed_rc)
    s = max(1, math.ceil(max(dem.shape) / n_max))
    d, w = dem[::s, ::s], water[::s, ::s]
    valid = np.isfinite(d)
    d = np.where(valid, d, float(np.nanmin(d)) if valid.any() else 0.0).astype(np.float32)
    ny, nx = d.shape
    url = TEXTURES_3D.get(texture)
    if url:
        tex = np.asarray(Image.fromarray(base_for_area(url, texture, tiles, nx * 2)).resize((nx, ny), RS.LANCZOS),
                         dtype=np.float32) / 255.0
    elif texture == "Höhenfarben":
        lo_, hi_ = float(np.percentile(d, 2)), float(np.percentile(d, 98)) + 1e-6
        tex = matplotlib.colormaps["terrain"](np.clip((d - lo_) / (hi_ - lo_), 0, 1))[..., :3].astype(np.float32)
    else:
        hs = hillshade(d, row_px(tiles, dem.shape[0])[::s][:ny] * s)
        tex = np.dstack([hs * 0.75 + 0.2] * 3).astype(np.float32)
    top = colorize(d, w, level, mode, alpha, shore).astype(np.float32) / 255.0
    ta = top[..., 3:4]
    col = (np.clip(top[..., :3] * ta + tex * (1 - ta), 0, 1) * 255 + 0.5).astype(np.uint8).reshape(-1, 3)
    z = np.where(w, np.float32(level), d)
    px_km = pixel_size_m(tiles[0]) * s * math.cos(math.radians((bounds[0] + bounds[2]) / 2)) / 1000
    xs, ys = np.arange(nx) * px_km, (ny - 1 - np.arange(ny)) * px_km
    r_, c_ = np.mgrid[0:ny - 1, 0:nx - 1]
    a = (r_ * nx + c_).ravel()
    b, cc = a + 1, a + nx
    return dict(X=np.tile(xs, ny), Y=np.repeat(ys, nx), Z=z.ravel(), i=np.concatenate([a, b]),
                j=np.concatenate([b, cc + 1]), k=np.concatenate([cc, cc]),
                vc=[f"rgb({r},{g},{bb})" for r, g, bb in col],
                xr=float(xs.max()) if nx > 1 else 1.0, yr=float(ys.max()) if ny > 1 else 1.0,
                zmin=float(z.min()), zmax=float(z.max()), n_wet=int(w.sum()), s=s)


def plotly_3d(sc, exag, height):
    fig = go.Figure(go.Mesh3d(
        x=sc["X"], y=sc["Y"], z=sc["Z"], i=sc["i"], j=sc["j"], k=sc["k"], vertexcolor=sc["vc"], flatshading=False,
        lighting=dict(ambient=0.75, diffuse=0.55, specular=0.04, roughness=0.9, fresnel=0.05),
        lightposition=dict(x=-1e5, y=1e5, z=1e5),
        hovertemplate="Ost %{x:.1f} km · Nord %{y:.1f} km<br>Höhe / Wasserspiegel %{z:.0f} m<extra></extra>"))
    xr, yr = max(sc["xr"], 1e-3), max(sc["yr"], 1e-3)
    m = max(xr, yr)
    zr = max(sc["zmax"] - sc["zmin"], 1.0) / 1000 * exag
    fig.update_layout(height=int(height), margin=dict(l=0, r=0, t=10, b=0), uirevision="3d",
                      scene=dict(aspectmode="manual", aspectratio=dict(x=xr / m, y=yr / m, z=max(zr / m, 0.02)),
                                 xaxis_title="Ost (km)", yaxis_title="Nord (km)", zaxis_title="Höhe (m)",
                                 camera=dict(eye=dict(x=1.1, y=-1.5, z=0.9))))
    return fig


# ============================== Burgen (OpenStreetMap) ==============================
@st.cache_data(show_spinner=False, max_entries=60)
def fetch_castles(cc):
    q = f"""[out:json][timeout:300];
area["ISO3166-1"="{cc}"]["admin_level"="2"]->.a;
(
  nwr["historic"="castle"]["castle_type"!~"^(palace|stately|manor|schloss|fortress)$"](area.a);
  nwr["historic"="ruins"]["ruins"="castle"](area.a);
);
out center tags;"""
    js, last = None, None
    for url in OVERPASS_URLS:
        try:
            r = requests.post(url, data={"data": q}, headers={"User-Agent": USER_AGENT}, timeout=330)
            r.raise_for_status()
            js = r.json()
            break
        except Exception as e:
            last = e
    if js is None:
        raise RuntimeError(f"Overpass nicht erreichbar oder Zeitüberschreitung ({last})")
    out, seen = [], set()
    for el in js.get("elements", []):
        tg = el.get("tags") or {}
        ct = (tg.get("castle_type") or "").lower()
        if ct in CASTLE_EXCLUDE:
            continue
        lat = el.get("lat", (el.get("center") or {}).get("lat"))
        lon = el.get("lon", (el.get("center") or {}).get("lon"))
        cid = f"{el['type'][0]}{el['id']}"
        if lat is None or lon is None or cid in seen:
            continue
        seen.add(cid)
        out.append(dict(id=cid, name=tg.get("name:de") or tg.get("name") or "(ohne Namen)", typ=ct,
                        ruine=tg.get("ruins") == "yes" or tg.get("historic") == "ruins", land=cc,
                        lat=round(float(lat), 6), lon=round(float(lon), 6), wikipedia=tg.get("wikipedia", ""),
                        wikidata=tg.get("wikidata", ""), baujahr=tg.get("start_date", "")))
    return out


def _flat(c):
    if len(c) and isinstance(c[0], (int, float)):
        return [c[:2]]
    return [p for s in c for p in _flat(s)]


def parse_castle_geojson(b):
    out = {}
    for i, f in enumerate(json.loads(b).get("features", [])):
        g, p = f.get("geometry") or {}, f.get("properties") or {}
        try:
            pts = np.asarray(_flat(g.get("coordinates") or []), float)
        except Exception:
            continue
        if not len(pts):
            continue
        ct = str(p.get("typ") or p.get("castle_type") or "")
        if ct.lower() in CASTLE_EXCLUDE:
            continue
        ru = p.get("ruine")
        if ru is None:
            ru = p.get("ruins") == "yes" or p.get("historic") == "ruins"
        cid = str(p.get("id") or p.get("@id") or f"u{i}")
        if "/" in cid:
            t_, n_ = cid.split("/", 1)
            cid = f"{t_[:1]}{n_}"
        cc = str(p.get("land") or "XX")
        out.setdefault(cc, []).append(dict(
            id=cid, name=p.get("name:de") or p.get("name") or "(ohne Namen)", typ=ct, ruine=bool(ru), land=cc,
            lat=round(float(pts[:, 1].mean()), 6), lon=round(float(pts[:, 0].mean()), 6),
            wikipedia=p.get("wikipedia") or "", wikidata=p.get("wikidata") or "",
            baujahr=str(p.get("baujahr") or p.get("start_date") or "")))
    return out


def merge_castles(add):
    for cc, lst in add.items():
        have = {c["id"]: c for c in S.castles.get(cc, [])}
        for c in lst:
            have[c["id"]] = c
        S.castles[cc] = list(have.values())


def castles_df():
    rows = [c for lst in S.castles.values() for c in lst]
    if not rows:
        return pd.DataFrame(columns=CASTLE_COLS)
    return pd.DataFrame(rows).drop_duplicates("id").reset_index(drop=True)


def castles_geojson(df):
    feats = []
    for r in df.to_dict("records"):
        props = {k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                 for k, v in r.items() if k not in ("lat", "lon")}
        feats.append({"type": "Feature", "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
                      "properties": props})
    return json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False,
                      default=lambda o: o.item() if hasattr(o, "item") else str(o)).encode("utf-8")


def wiki_url(w):
    if not w or ":" not in str(w):
        return ""
    lang, title = str(w).split(":", 1)
    return f"https://{lang}.wikipedia.org/wiki/{title.replace(' ', '_')}"


@st.cache_resource
def _castle_store():
    return {}


def _tile_raw(z, x, y):
    try:
        r = _session.get(TERRARIUM_URL.format(z=z, x=x, y=y), timeout=20)
        r.raise_for_status()
        return _decode_terrarium(r.content)
    except Exception:
        return None


def compute_castle_heights(df, z, radius_m, prog=None):
    """Höhe jeder Burg + tiefster Punkt im Umkreis (für 'am Ufer') + Zufallspunkte als Vergleich."""
    store = _castle_store().setdefault((z, int(radius_m)), {"elev": {}, "near": {}, "rand": [], "tiles": set()})
    todo = df[~df["id"].isin(store["elev"].keys())]
    if not len(todo):
        return store
    n = 2 ** z
    x, y = lonlat_to_tile(todo["lon"].to_numpy(float), todo["lat"].to_numpy(float), z)
    tx, ty = np.floor(x).astype(int), np.floor(y).astype(int)
    px = np.clip(((x - tx) * TILE).astype(int), 0, TILE - 1)
    py = np.clip(((y - ty) * TILE).astype(int), 0, TILE - 1)
    ids, lats = todo["id"].to_numpy(), todo["lat"].to_numpy(float)
    groups = {}
    for i, k in enumerate(zip(tx.tolist(), ty.tolist())):
        groups.setdefault(k, []).append(i)
    keys = list(groups)
    with ThreadPoolExecutor(max_workers=12) as ex:
        for b0 in range(0, len(keys), 150):
            batch = keys[b0:b0 + 150]
            for k, t in zip(batch, ex.map(lambda k_: _tile_raw(z, k_[0] % n, k_[1]), batch)):
                idx = groups[k]
                if t is None:
                    for i in idx:
                        store["elev"][ids[i]] = store["near"][ids[i]] = np.nan
                    continue
                px_m = max(pixel_size_m(z) * math.cos(math.radians(float(lats[idx[0]]))), 1.0)
                rp = max(1, int(math.ceil(radius_m / px_m)))
                for i in idx:
                    r, c = int(py[i]), int(px[i])
                    store["elev"][ids[i]] = float(t[r, c])
                    store["near"][ids[i]] = float(t[max(0, r - rp):r + rp + 1, max(0, c - rp):c + rp + 1].min())
                if k not in store["tiles"]:
                    store["tiles"].add(k)
                    rng = np.random.default_rng(abs(hash((z,) + k)) % (2 ** 32))
                    for r, c in rng.integers(0, TILE, size=(4, 2)):
                        lo_, la_ = tile_to_lonlat(k[0] + (c + 0.5) / TILE, k[1] + (r + 0.5) / TILE, z)
                        store["rand"].append((float(la_), float(lo_), float(t[r, c]),
                                              float(t[max(0, r - rp):r + rp + 1, max(0, c - rp):c + rp + 1].min())))
            if prog is not None:
                done = min(b0 + len(batch), len(keys))
                prog.progress(done / len(keys), text=f"Lade Höhendaten für Burgen … {done}/{len(keys)} Kacheln")
    return store


def castle_arrays(cdf, z, r):
    store = _castle_store().get((int(z), int(r)))
    if not store or cdf.empty:
        nanv = np.full(len(cdf), np.nan)
        return nanv, nanv.copy()
    return cdf["id"].map(store["elev"]).to_numpy(float), cdf["id"].map(store["near"]).to_numpy(float)


def castle_status(elev, near, level):
    out = np.full(len(elev), "unbekannt", dtype=object)
    ok = np.isfinite(elev)
    with np.errstate(invalid="ignore"):
        under = ok & (elev < level)
        shore_ = ok & ~under & (near < level)
    out[under], out[shore_] = "unter Wasser", "am Ufer"
    out[ok & ~under & ~shore_] = "trocken"
    return out


def in_view(lat, lon, v):
    return (lat >= v[0]) & (lat <= v[2]) & (lon >= v[1]) & (lon <= v[3])


def castle_shore_curve(e, nm, re_, rn, levels):
    ok = np.isfinite(e)
    e, nm = e[ok], nm[ok]
    L = levels[:, None]
    dry = e[None, :] >= L
    cu, cd, cw = (dry & (nm[None, :] < L)).sum(1), dry.sum(1), (~dry).sum(1)
    rok = np.isfinite(re_)
    re_, rn = re_[rok], rn[rok]
    rdry = re_[None, :] >= L
    ru, rd = (rdry & (rn[None, :] < L)).sum(1), rdry.sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cf = np.where(cd > 0, cu / np.maximum(cd, 1) * 100, np.nan)
        rf = np.where(rd > 0, ru / np.maximum(rd, 1) * 100, np.nan)
        fac = np.where(rf > 0, cf / rf, np.nan)
    return pd.DataFrame({"wasserstand_m": levels, "burgen_am_ufer": cu, "burgen_unter_wasser": cw,
                         "anteil_burgen_pct": cf, "anteil_zufall_pct": rf, "faktor": fac})


def castle_popup(r, stt, ev, level):
    nm = html.escape(str(r["name"]))
    s = f"<b>🏰 {nm}</b>{' (Ruine)' if r['ruine'] is True else ''}<br>Zustand: <b>{stt}</b>"
    if np.isfinite(ev):
        s += f"<br>Höhe: {ev:.0f} m · über Wasser: {ev - level:+.0f} m"
    if r.get("baujahr"):
        s += f"<br>Baujahr (OSM): {html.escape(str(r['baujahr']))}"
    u = wiki_url(r.get("wikipedia"))
    if u:
        s += f'<br><a href="{html.escape(u)}" target="_blank">Wikipedia</a>'
    s += f"<br><small>{r['lat']:.4f}, {r['lon']:.4f}</small>"
    return s


def add_castle_layers(fg):
    cdf = castles_df()
    if cdf.empty:
        return
    e, nm = castle_arrays(cdf, S.castle_z, S.castle_r)
    status = castle_status(e, nm, float(S.level))
    lv = float(S.level)
    for stt, col in CASTLE_COLORS.items():
        if stt == "trocken" and not S.castle_dry:
            continue
        sel = status == stt
        if not sel.any():
            continue
        sub, es = cdf[sel], e[sel]
        data = [[float(r["lat"]), float(r["lon"]), f"🏰 {html.escape(str(r['name']))}", castle_popup(r, stt, ev, lv)]
                for r, ev in zip(sub.to_dict("records"), es)]
        plugins.FastMarkerCluster(data, callback=CASTLE_CB.replace("__C__", col),
                                  icon_create_function=CASTLE_ICON.replace("__C__", col),
                                  name=f"Burgen {stt}", control=False).add_to(fg)


def load_countries(names):
    prog = st.progress(0.0, text="Lade Burgen …")
    errs = []
    for i, nm in enumerate(names):
        prog.progress(i / max(len(names), 1), text=f"Lade Burgen: {nm} ({i + 1}/{len(names)}) …")
        try:
            merge_castles({EU_COUNTRIES[nm]: fetch_castles(EU_COUNTRIES[nm])})
        except Exception as e:
            errs.append(f"{nm}: {e}")
    prog.empty()
    return errs


# ============================== Hilfsfunktionen ==============================
def de_num(v, nd=1, sign=False):
    s = f"{v:{'+' if sign else ''},.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def approx_view(center, zoom, w=1200, h=MAP_HEIGHT):
    x, y = lonlat_to_tile(center[1], center[0], zoom)
    dx, dy = w / 2 / TILE, h / 2 / TILE
    lon_w, lat_n = tile_to_lonlat(x - dx, y - dy, zoom)
    lon_e, lat_s = tile_to_lonlat(x + dx, y + dy, zoom)
    return (round(float(lat_s), 4), round(float(lon_w), 4), round(float(lat_n), 4), round(float(lon_e), 4), int(zoom))


def view_from(out):
    if not out or not out.get("bounds"):
        return None
    sw, ne = out["bounds"].get("_southWest"), out["bounds"].get("_northEast")
    if not sw or not ne or sw.get("lat") is None or ne.get("lat") is None:
        return None
    return (round(sw["lat"], 4), round(sw["lng"], 4), round(ne["lat"], 4), round(ne["lng"], 4),
            int(out.get("zoom") or S.zoom))


def set_level(v):
    v = float(np.clip(v, LEVEL_MIN, LEVEL_MAX))
    S.level, S.level_input, S.level_slider = v, v, v


def shift_level(d):
    set_level(S.level + d)


def go_to(center, zoom, level=None):
    S.center, S.zoom = [float(center[0]), float(center[1])], int(zoom)
    S.view = S.calc_view = approx_view(S.center, S.zoom)
    if level is not None:
        set_level(level)


def calc_now():
    S.calc_view = S.view


def anim_around():
    S.anim_lo, S.anim_hi = float(max(LEVEL_MIN, S.level - 100)), float(min(LEVEL_MAX, S.level + 100))


@st.cache_data(show_spinner=False, max_entries=100)
def geocode(q):
    r = requests.get("https://nominatim.openstreetmap.org/search", params=dict(q=q, format="json", limit=1),
                     headers={"User-Agent": USER_AGENT}, timeout=15)
    r.raise_for_status()
    j = r.json()
    return (float(j[0]["lat"]), float(j[0]["lon"]), j[0].get("display_name", "")) if j else None


def markers_geojson():
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [m["lon"], m["lat"]]},
              "properties": {k: v for k, v in m.items() if k not in ("lat", "lon")}} for m in S.markers]
    return json.dumps({"type": "FeatureCollection", "features": feats}, ensure_ascii=False, indent=1).encode()


def build_map():
    m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles=None, world_copy_jump=True,
                   control_scale=True, min_zoom=2)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(tiles=ESRI_URL, attr="Esri World Imagery", name="Satellit (Esri)").add_to(m)
    folium.TileLayer(tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                     attr="© OpenTopoMap (CC-BY-SA), © OpenStreetMap-Mitwirkende",
                     name="Gelände (OpenTopoMap)", max_zoom=17).add_to(m)
    plugins.MarkerCluster(control=False).add_to(m)   # lädt die Gruppierungs-Bibliothek
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MeasureControl(position="topleft", primary_length_unit="kilometers",
                           primary_area_unit="sqkilometers").add_to(m)
    plugins.Draw(export=False, position="topleft",
                 draw_options={"polyline": {"shapeOptions": {"color": "#e41a1c", "weight": 3}}, "polygon": False,
                               "rectangle": False, "circle": False, "marker": False, "circlemarker": False},
                 edit_options={"edit": False, "remove": True}).add_to(m)
    plugins.MousePosition(position="bottomleft", separator=" · ", num_digits=4, prefix="Lat/Lon:").add_to(m)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)
    folium.LayerControl(position="topright").add_to(m)
    return m


# Gespeicherte Burgen-Sammlung aus dem Repo automatisch laden
if not S.castles and not S.get("_castle_file_done"):
    S["_castle_file_done"] = True
    try:
        _cf = os.path.join(os.path.dirname(os.path.abspath(__file__)), CASTLE_FILE)
        if os.path.exists(_cf):
            with open(_cf, "rb") as _fh:
                merge_castles(parse_castle_geojson(_fh.read()))
    except Exception:
        pass

# ============================== Sidebar ==============================
with st.sidebar:
    st.markdown("## 🌊 Wasserstand")
    st.metric("Aktueller Wasserstand", f"{de_num(S.level, sign=True)} m")
    st.slider("Grobeinstellung (m)", LEVEL_MIN, LEVEL_MAX, step=10.0, key="level_slider",
              on_change=lambda: set_level(S.level_slider))
    st.number_input("Genauer Wert (m)", LEVEL_MIN, LEVEL_MAX, step=1.0, format="%.1f", key="level_input",
                    on_change=lambda: set_level(S.level_input))
    bc = st.columns(6)
    for i, d in enumerate([-100, -10, -1, 1, 10, 100]):
        bc[i].button(f"{d:+d}", key=f"lv_{d}", on_click=shift_level, args=(float(d),), use_container_width=True)
    for label, lv in PRESETS:
        st.button(label, key=f"pre_{lv}", on_click=set_level, args=(lv,), use_container_width=True)

    with st.expander("🧭 Flutungsmodus", expanded=True):
        st.radio("Wie breitet sich das Wasser aus?", FLOOD_MODES, key="flood_mode")
        st.checkbox("Klick-Abfrage aktiv", key="click_on", help="Zum Zeichnen von Profillinien ausschalten.")

    with st.expander("🏰 Burgen"):
        _nc = sum(len(v) for v in S.castles.values())
        st.caption(f"{_nc:,} Burgen geladen".replace(",", ".") if _nc else "Noch keine Burgen – Reiter „🏰 Burgen“.")
        st.checkbox("Burgen auf der Karte zeigen", key="castle_map")
        st.checkbox("Trockene Burgen anzeigen", key="castle_dry")

    with st.expander("🎨 Darstellung"):
        st.radio("Ansicht", MODES, key="mode")
        st.slider("Deckkraft Wasser", 0.2, 1.0, step=0.05, key="alpha")
        st.checkbox("Uferlinie", key="shore")
        st.checkbox("Relief-Schummerung", key="hill_on")
        st.slider("Deckkraft Relief", 0.1, 1.0, step=0.05, key="hill_op", disabled=not S.hill_on)
        st.checkbox("Abflusslose Senken (lila)", key="sinks_on", disabled=not SKIMAGE_OK)
        st.slider("Mindesttiefe Senken (m)", 0.5, 50.0, step=0.5, key="sinks_min", disabled=not S.sinks_on)
        st.checkbox("Automatisch aktualisieren", key="auto")
        if not S.auto:
            st.button("🔄 Ausschnitt berechnen", on_click=calc_now, use_container_width=True, type="primary")

    with st.expander("⚖️ Vergleich & Unsicherheit"):
        st.checkbox("Zweiter Wasserstand als rote Linie", key="l2_on")
        st.number_input("Zweiter Wasserstand (m)", LEVEL_MIN, LEVEL_MAX, step=10.0, key="level2", disabled=not S.l2_on)
        st.checkbox("Unsicherheitszone anzeigen (gelb)", key="band_on")
        st.slider("± Meter", 1.0, 200.0, step=1.0, key="band_m", disabled=not S.band_on)

    with st.expander("🧩 Eigene Kartenebenen"):
        st.text_input("WMS-URL", key="wms_url", placeholder="https://…/wms?")
        st.text_input("WMS-Layername(n)", key="wms_layers")
        st.text_input("XYZ-URL", key="xyz_url", placeholder="https://…/{z}/{x}/{y}.png")
        st.slider("Deckkraft eigene Ebenen", 0.1, 1.0, step=0.05, key="ovl_op")

    with st.expander("📍 Ort", expanded=True):
        with st.form("suche"):
            q = st.text_input("Ort suchen", placeholder="z. B. Mainz, Great Salt Lake")
            submitted = st.form_submit_button("🔍 Suchen", use_container_width=True)
        if submitted and q.strip():
            try:
                res = geocode(q.strip())
            except Exception as e:
                st.error(f"Suche fehlgeschlagen: {e}")
            else:
                if res:
                    go_to([res[0], res[1]], 10)
                    st.success(res[2][:90])
                else:
                    st.warning("Nichts gefunden.")
        st.selectbox("Beispielorte", list(PLACES), key="place")
        st.button("➡️ Hinspringen", use_container_width=True, on_click=lambda: go_to(**PLACES[S.place]))
    st.caption(f"Flut-App · Version {APP_VERSION}")

# ============================== Hauptbereich ==============================
st.markdown("""<div class="app-header"><h1>🌊 Flut-App</h1>
<p>Meeresspiegel und Wasserstände weltweit simulieren – von −11 000 m bis +5000 m</p></div>""", unsafe_allow_html=True)

if S.view is None:
    S.view = approx_view(S.center, S.zoom)
calc_view = S.view if S.auto else (S.calc_view or S.view)
bounds = normalize_bounds(*calc_view[:4])
tiles = choose_tiles(bounds, calc_view[4])
lvl = round(float(S.level), 1)
seed_rc = None
if S.flood_mode == FLOOD_MODES[2] and S.click:
    _r, _c = lonlat_to_pixel(tiles, S.click["lat"], S.click["lon"])
    seed_rc = (int(_r), int(_c))
level2 = round(float(S.level2), 1) if S.l2_on else None
band = float(S.band_m) if S.band_on else None

t_map, t_3d, t_burg, t_prof, t_terr, t_anim, t_mark, t_exp, t_help = st.tabs(
    ["🗺️ Karte", "🏔️ 3D", "🏰 Burgen", "📈 Profil", "🏖️ Strandlinien", "🎬 Animation", "📍 Markierungen",
     "📦 Export", "❓ Hilfe"])

# ---------------- Karte ----------------
with t_map:
    fg = folium.FeatureGroup(name="Flut-Ebenen")
    info = None
    try:
        info = water_layer(tiles, lvl, S.mode, S.flood_mode, seed_rc, round(float(S.alpha), 2), bool(S.shore),
                           level2, band, bool(S.hill_on))
        ib = info["bounds"]
        box = [[ib[0], ib[1]], [ib[2], ib[3]]]
        if info["hill_png"]:
            folium.raster_layers.ImageOverlay(to_uri(info["hill_png"]), bounds=box, opacity=float(S.hill_op),
                                              interactive=False, zindex=5).add_to(fg)
        if S.sinks_on and SKIMAGE_OK:
            folium.raster_layers.ImageOverlay(to_uri(sinks_layer(tiles, float(S.sinks_min))), bounds=box,
                                              opacity=1.0, interactive=False, zindex=8).add_to(fg)
        folium.raster_layers.ImageOverlay(to_uri(info["png"]), bounds=box, opacity=1.0, interactive=False,
                                          zindex=10).add_to(fg)
    except Exception as e:
        st.error(f"Höhendaten konnten nicht geladen werden: {e}")
    if S.wms_url.strip() and S.wms_layers.strip():
        folium.raster_layers.WmsTileLayer(url=S.wms_url.strip(), layers=S.wms_layers.strip(), fmt="image/png",
                                          transparent=True, name="WMS", overlay=True,
                                          opacity=float(S.ovl_op)).add_to(fg)
    if S.xyz_url.strip():
        folium.TileLayer(tiles=S.xyz_url.strip(), attr="Eigene Ebene", name="XYZ", overlay=True,
                         opacity=float(S.ovl_op)).add_to(fg)
    if S.castle_map and S.castles:
        try:
            add_castle_layers(fg)
        except Exception as e:
            st.warning(f"Burgen konnten nicht angezeigt werden: {e}")
    for mk in S.markers:
        folium.Marker([mk["lat"], mk["lon"]], tooltip=mk["name"],
                      popup=folium.Popup(f"<b>{mk['name']}</b><br>{mk.get('notiz', '')}<br>Höhe: "
                                         f"{mk.get('hoehe_m', '–')} m", max_width=260),
                      icon=folium.Icon(color="green", icon="star")).add_to(fg)
    if S.click:
        folium.Marker([S.click["lat"], S.click["lon"]], tooltip="Abfragepunkt",
                      icon=folium.Icon(color="red", icon="info-sign")).add_to(fg)
    if S.drawn_line:
        folium.PolyLine([[b, a] for a, b in S.drawn_line], color="#e41a1c", weight=3, tooltip="Profillinie").add_to(fg)

    out = st_folium(build_map(), key="karte", height=MAP_HEIGHT, use_container_width=True, center=S.center,
                    zoom=S.zoom, feature_group_to_add=fg,
                    returned_objects=["bounds", "zoom", "last_clicked", "all_drawings"])

    need_rerun = False
    new_view = view_from(out)
    if new_view and new_view != S.view:
        S.view = new_view
        if S.auto and choose_tiles(normalize_bounds(*new_view[:4]), new_view[4]) != tiles:
            need_rerun = True
    lines = [f for f in ((out or {}).get("all_drawings") or [])
             if f and (f.get("geometry") or {}).get("type") == "LineString"]
    if lines:
        coords = [[round(float(a), 5), round(float(b), 5)] for a, b in lines[-1]["geometry"]["coordinates"]]
        if len(coords) >= 2 and coords != S.drawn_line:
            S.drawn_line = coords
    lc = (out or {}).get("last_clicked")
    if S.click_on and lc and lc.get("lat") is not None:
        key = (round(lc["lat"], 5), round(lc["lng"], 5))
        if not S.click or (S.click["lat"], S.click["lon"]) != key:
            zc = min(MAX_TILE_ZOOM, (S.view[4] if S.view else S.zoom) + 2)
            S.click = dict(lat=key[0], lon=key[1], elev=elevation_at(key[0], key[1], zc))
            need_rerun = True
    if need_rerun and S._rr < 3:
        S._rr += 1
        st.rerun()
    S._rr = 0

    leg = ("<span style='background:#7cc8f8'></span>flach<span style='background:#1f78d1'></span>mittel"
           "<span style='background:#08306b'></span>tief") if S.mode == MODES[0] else \
          ("<span style='background:#1e78dc'></span>neu überflutet<span style='background:#e0aa46'></span>"
           "trockengefallen<span style='background:#96bee6'></span>auch heute Meer")
    leg += "<span style='background:#ffd700'></span>Uferlinie"
    if level2 is not None:
        leg += f"<span style='background:#dc2828'></span>Vergleich {de_num(level2, 0, True)} m"
    if band:
        leg += f"<span style='background:#ffeb78'></span>Unsicherheit ±{band:.0f} m"
    if S.sinks_on:
        leg += "<span style='background:#963cc8'></span>abflusslose Senke"
    if S.castle_map and S.castles:
        leg += ("<span style='background:#d62828'></span>Burg unter Wasser<span style='background:#1f78d1'></span>"
                "Burg am Ufer<span style='background:#8a8a8a'></span>Burg trocken"
                "<span style='background:#8b5a2b'></span>Höhe nicht berechnet")
    st.markdown(f"<div class='legend'>{leg}</div>", unsafe_allow_html=True)

    if info:
        stt = info["stats"]
        res_m = pixel_size_m(tiles[0]) * math.cos(math.radians((bounds[0] + bounds[2]) / 2))
        r1 = st.columns(4)
        r1[0].metric("Wasserstand", f"{de_num(S.level, sign=True)} m")
        r1[1].metric("Wasserfläche im Ausschnitt", f"{de_num(stt['water'], 0)} km²")
        r1[2].metric("Neu überflutet", f"{de_num(stt['new_flood'], 0)} km²")
        r1[3].metric("Trockengefallen", f"{de_num(stt['new_dry'], 0)} km²")
        r2 = st.columns(4)
        r2[0].metric("Wasservolumen im Ausschnitt", f"{de_num(stt['volume'], 1)} km³")
        r2[1].metric("Getrennte Wasserflächen", f"{stt['bodies']:,}".replace(",", "."))
        r2[2].metric("Inseln", f"{stt['islands']:,}".replace(",", "."), f"{de_num(stt['isl_area'], 0)} km²",
                     delta_color="off")
        r2[3].metric("Auflösung Höhendaten", f"≈ {de_num(res_m, 0)} m", f"Kachelzoom {tiles[0]}", delta_color="off")
        if info["missing"]:
            st.warning(f"{info['missing']} Höhenkachel(n) fehlen – dort ist kein Wasser dargestellt.")
        if S.flood_mode == FLOOD_MODES[2] and not S.click:
            st.info("Modus „Ab Startpunkt“: Klicke auf die Karte, um den Startpunkt zu setzen.")

    if S.click:
        c = S.click
        st.markdown("#### 📍 Abfragepunkt")
        k = st.columns(4)
        k[0].metric("Koordinaten", f"{c['lat']:.4f}, {c['lon']:.4f}")
        if c["elev"] is None:
            k[1].metric("Geländehöhe", "keine Daten")
        else:
            k[1].metric("Geländehöhe", f"{de_num(c['elev'], 1, sign=True)} m")
            diff = S.level - c["elev"]
            k[2].metric("Zustand", "unter Wasser" if diff > 0 else "trocken",
                        f"Tiefe {de_num(diff, 1)} m" if diff > 0 else f"{de_num(-diff, 1)} m über dem Wasser",
                        delta_color="off")
            if SKIMAGE_OK:
                try:
                    r, cc = lonlat_to_pixel(tiles, c["lat"], c["lon"])
                    fd = filled_dem(tiles)
                    r, cc = int(r), int(cc)
                    if 0 <= r < fd.shape[0] and 0 <= cc < fd.shape[1]:
                        spill = float(fd[r, cc])
                        if spill > c["elev"] + 0.5:
                            k[3].metric("Überlaufhöhe", f"{de_num(spill, 1, True)} m",
                                        f"Becken bis {de_num(spill - c['elev'], 1)} m tief", delta_color="off")
                        else:
                            k[3].metric("Überlaufhöhe", "kein Becken", "fließt zum Rand ab", delta_color="off")
                except Exception:
                    pass
        with st.form("marker_form", clear_on_submit=True):
            mc1, mc2 = st.columns([1, 2])
            mname = mc1.text_input("Name der Markierung", value=f"Punkt {len(S.markers) + 1}")
            mnote = mc2.text_input("Notiz")
            if st.form_submit_button("⭐ Als Markierung speichern"):
                S.markers.append(dict(name=mname, notiz=mnote, lat=c["lat"], lon=c["lon"],
                                      hoehe_m=None if c["elev"] is None else round(c["elev"], 1),
                                      wasserstand_m=lvl, zeit=datetime.now().strftime("%d.%m.%Y %H:%M")))
                st.rerun()
    else:
        st.caption("💡 Klick auf die Karte = Höhe, Wassertiefe, Überlaufhöhe. Klick auf eine Burg = Info-Fenster. "
                   "Werkzeuge oben links: Vollbild, Messen, Linie zeichnen (Profil).")

    with st.expander("💧 Wasserbilanz – wie viel Wasser bräuchte dieser Meeresspiegel?"):
        dv = OCEAN_AREA_KM2 * S.level / 1000.0
        w1, w2, w3 = st.columns(3)
        w1.metric("Volumenänderung weltweit (grob)", f"{de_num(dv / 1e6, 1, True)} Mio. km³")
        w2.metric("Anteil am heutigen Ozeanvolumen", f"{de_num(dv / OCEAN_VOL_KM3 * 100, 2, True)} %")
        w3.metric("Vielfaches allen Eises", f"{de_num(S.level / ICE_SLE_M, 1, True)} ×")
        st.caption("Überschlag: Ozeanfläche × Wasserstand. Alles Eis ≈ 65–70 m Meeresspiegel. Nur Größenordnung.")

    with st.expander("📈 Flutkurve für den sichtbaren Ausschnitt"):
        try:
            fc = flood_curve(tiles)
            if len(fc):
                f = go.Figure(go.Scatter(x=fc.wasserstand_m, y=fc.flaeche_km2, mode="lines", line=dict(color="#1f6fb2")))
                f.add_vline(x=S.level, line_dash="dash", line_color="gray")
                f.update_layout(height=340, margin=dict(t=20, b=10), xaxis_title="Wasserstand (m)",
                                yaxis_title="Überflutete Fläche (km²)")
                st.plotly_chart(f, use_container_width=True)
        except Exception as e:
            st.warning(f"Flutkurve nicht verfügbar: {e}")

# ---------------- 3D ----------------
with t_3d:
    c1, c2, c3, c4 = st.columns(4)
    c1.selectbox("Oberfläche", TEX_EXACT, key="tex3e")
    c2.slider("Überhöhung", 1.0, 30.0, step=0.5, key="exag3e")
    c3.select_slider("Detailgrad (Punkte je Seite)", RES_3D, key="res3e")
    c4.slider("Deckkraft Wasserfarbe", 0.3, 1.0, step=0.05, key="walpha3e")
    try:
        sc3 = scene_3d(tiles, lvl, S.flood_mode, seed_rc, S.tex3e, int(S.res3e), S.mode,
                       round(float(S.walpha3e), 2), bool(S.shore))
        st.plotly_chart(plotly_3d(sc3, float(S.exag3e), MAP_HEIGHT), use_container_width=True,
                        config={"scrollZoom": True, "displaylogo": False})
        if sc3["n_wet"] == 0:
            st.info("Im aktuellen Ausschnitt steht bei diesem Wasserstand und Modus kein Wasser.")
        st.caption("🖱️ Links = Drehen · Rechts = Verschieben · Mausrad = Zoom · Doppelklick = zurücksetzen. Gleiche "
                   f"Wassermaske wie die 2D-Karte (3D-Raster: jeder {sc3['s']}. Punkt). Ausschnitt: 2D-Karte bewegen.")
    except Exception as e:
        st.error(f"3D-Ansicht konnte nicht erzeugt werden: {e}")

# ---------------- Burgen ----------------
with t_burg:
    st.caption("🏰 Burgen und Burgruinen aus OpenStreetMap – ohne Schlösser, Paläste, Herrenhäuser und Festungen. "
               "Gezählt wird, was in OSM eingetragen ist.")
    cdf_all = castles_df()
    with st.expander("📥 Burgen laden & verwalten", expanded=cdf_all.empty):
        l1, l2 = st.columns([3, 1])
        l1.multiselect("Länder", list(EU_COUNTRIES), key="castle_sel")
        l2.write("")
        go_sel = l2.button("📥 Ausgewählte laden", use_container_width=True, type="primary")
        go_all = st.button("🌍 Ganz Europa laden (dauert mehrere Minuten)")
        if go_sel or go_all:
            S["_castle_errs"] = load_countries(list(EU_COUNTRIES) if go_all else list(S.castle_sel))
            st.rerun()
        for err in S.get("_castle_errs") or []:
            st.warning(f"⚠️ {err} – später erneut versuchen oder per overpass-turbo exportieren und hochladen.")
        if S.castles:
            st.write("**Geladen:** " + " · ".join(f"{CC_NAME.get(cc, cc)} {len(v):,}".replace(",", ".")
                                                   for cc, v in sorted(S.castles.items())))
        u1, u2 = st.columns(2)
        with u1:
            up = st.file_uploader("Sammlung oder eigene Liste laden (GeoJSON)", type=["geojson", "json"], key="castle_up")
            if up is not None and st.button("Hochgeladene Datei übernehmen"):
                try:
                    merge_castles(parse_castle_geojson(up.getvalue()))
                    st.rerun()
                except Exception as e:
                    st.error(f"Datei konnte nicht gelesen werden: {e}")
        with u2:
            if not cdf_all.empty:
                st.download_button("💾 Sammlung speichern (burgen_europa.geojson)", castles_geojson(cdf_all),
                                   CASTLE_FILE, "application/geo+json", use_container_width=True)
                st.caption("Datei ins GitHub-Repo neben app.py legen → wird beim Start automatisch geladen.")
                if st.button("🗑️ Sammlung leeren"):
                    S.castles = {}
                    st.rerun()

    if cdf_all.empty:
        st.info("Noch keine Burgen geladen. Oben Länder auswählen und „Ausgewählte laden“ klicken.")
    else:
        s1, s2, s3 = st.columns(3)
        s1.radio("Gebiet", CASTLE_AREAS, key="castle_area")
        s2.select_slider("„Am Ufer“ = Wasser im Umkreis von", [250, 500, 1000, 2000], key="castle_r",
                         format_func=lambda v: f"{v} m")
        s3.select_slider("Genauigkeit Burghöhen", list(CASTLE_Z), key="castle_z", format_func=lambda z: CASTLE_Z[z])
        vv = S.view or approx_view(S.center, S.zoom)
        cdf = cdf_all if S.castle_area == CASTLE_AREAS[0] else \
            cdf_all[in_view(cdf_all["lat"].to_numpy(float), cdf_all["lon"].to_numpy(float), vv)].reset_index(drop=True)
        e, nm = castle_arrays(cdf, S.castle_z, S.castle_r)
        n_miss = int((~np.isfinite(e)).sum()) if len(cdf) else 0
        if n_miss:
            st.warning(f"Für {n_miss:,} von {len(cdf):,} Burgen fehlen noch die Höhen.".replace(",", "."))
            if st.button("📏 Höhen berechnen", type="primary"):
                prog = st.progress(0.0, text="Lade Höhendaten für Burgen …")
                compute_castle_heights(cdf, int(S.castle_z), int(S.castle_r), prog)
                prog.empty()
                st.rerun()
        status = castle_status(e, nm, float(S.level))
        ok = np.isfinite(e)
        if len(cdf) == 0:
            st.info("Im gewählten Gebiet liegen keine geladenen Burgen.")
        elif ok.any():
            n_ok = int(ok.sum())
            cnt = {k_: int((status == k_).sum()) for k_ in CASTLE_COLORS}
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Burgen im Gebiet", f"{n_ok:,}".replace(",", "."), f"Wasserstand {de_num(S.level, 0, True)} m",
                      delta_color="off")
            for col_, k_ in zip((m2, m3, m4), ("unter Wasser", "am Ufer", "trocken")):
                col_.metric(k_.capitalize(), f"{cnt[k_]:,}".replace(",", "."), f"{de_num(cnt[k_] / n_ok * 100, 1)} %",
                            delta_color="off")
            store = _castle_store().get((int(S.castle_z), int(S.castle_r))) or {"rand": []}
            rand = np.asarray(store["rand"], float).reshape(-1, 4)
            if S.castle_area == CASTLE_AREAS[1] and len(rand):
                rand = rand[in_view(rand[:, 0], rand[:, 1], vv)]
            ev = e[ok]
            lo_ = max(LEVEL_MIN, float(np.percentile(ev, 1)) - 20)
            levels = np.linspace(lo_, max(float(np.percentile(ev, 99)) + 20, lo_ + 10), 300)
            cur = castle_shore_curve(e, nm, rand[:, 2] if len(rand) else np.array([]),
                                     rand[:, 3] if len(rand) else np.array([]), levels)
            st.markdown("#### 📈 Uferkurve – bei welchem Wasserstand liegen besonders viele Burgen am Ufer?")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cur.wasserstand_m, y=cur.burgen_am_ufer, name="Burgen am Ufer (Anzahl)",
                                 marker_color="rgba(31,120,209,0.35)", yaxis="y2"))
            fig.add_trace(go.Scatter(x=cur.wasserstand_m, y=cur.anteil_burgen_pct, name="Burgen am Ufer (%)",
                                     line=dict(color="#1f78d1", width=3)))
            fig.add_trace(go.Scatter(x=cur.wasserstand_m, y=cur.anteil_zufall_pct, name="Zufallspunkte am Ufer (%)",
                                     line=dict(color="#888", dash="dot", width=2)))
            fig.add_vline(x=S.level, line_dash="dash", line_color="#d62828")
            fig.update_layout(height=420, margin=dict(t=20, b=10), bargap=0, xaxis_title="Wasserstand (m)",
                              yaxis=dict(title="Anteil trockener Punkte am Ufer (%)"),
                              yaxis2=dict(title="Anzahl Burgen", overlaying="y", side="right", showgrid=False),
                              legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, use_container_width=True)
            fac, cu = cur.faktor.to_numpy(), cur.burgen_am_ufer.to_numpy()
            minc = max(5, int(0.02 * n_ok))
            cand = [i for i in range(1, len(fac) - 1) if np.isfinite(fac[i]) and cu[i] >= minc and fac[i] >= 1.3
                    and fac[i] >= np.nanmax(fac[max(0, i - 5):i + 6])]
            top = sorted(cand, key=lambda i: -fac[i])[:5]
            if top:
                st.write("**Auffällige Wasserstände** (Burgen deutlich häufiger am Ufer als Zufallspunkte) – "
                         "Klick setzt den Wasserstand:")
                pcs = st.columns(len(top))
                for j, i in enumerate(top):
                    h = float(cur.wasserstand_m.iloc[i])
                    pcs[j].button(f"{h:.0f} m · {int(cu[i])} Burgen · ×{fac[i]:.1f}", key=f"cpk_{j}",
                                  on_click=set_level, args=(h,), use_container_width=True)
            else:
                st.info("Keine Wasserstände, bei denen Burgen deutlich häufiger am Ufer liegen als Zufallspunkte (×1,3).")
            st.caption(f"Blau = Burgen, Grau = Zufallspunkte aus denselben Gebieten ({len(rand)} Punkte). Nur wenn Blau "
                       "deutlich über Grau liegt, ist die Häufung mehr als Zufall. Bei 0 m liegen viele Burgen an Küsten "
                       "und Flüssen – das ist bekannt. Burgen unter Wasser zeigen: Dieser Wasserstand kann zu ihrer "
                       "Bauzeit nicht bestanden haben.")

            st.markdown("#### 📋 Burgenliste")
            st.multiselect("Zustand anzeigen", list(CASTLE_COLORS), key="castle_filter")
            tdf = cdf.assign(Zustand=status, Hoehe_m=np.round(e, 0), Ueber_Wasser_m=np.round(e - S.level, 0),
                             Land=cdf["land"].map(lambda c_: CC_NAME.get(c_, c_)),
                             Wikipedia=cdf["wikipedia"].map(wiki_url))
            tdf = tdf[tdf["Zustand"].isin(S.castle_filter)].sort_values("Ueber_Wasser_m")
            cols = ["name", "Zustand", "Hoehe_m", "Ueber_Wasser_m", "ruine", "typ", "baujahr", "Land", "lat", "lon",
                    "Wikipedia"]
            st.dataframe(tdf[cols].head(3000), use_container_width=True, hide_index=True,
                         column_config={"Wikipedia": st.column_config.LinkColumn(display_text="Öffnen"),
                                        "name": "Name", "ruine": "Ruine", "typ": "Typ", "baujahr": "Baujahr",
                                        "Hoehe_m": "Höhe (m)", "Ueber_Wasser_m": "Über Wasser (m)"})
            if len(tdf) > 3000:
                st.caption(f"Angezeigt: 3.000 von {len(tdf):,} – vollständig im Export.".replace(",", "."))
            x1, x2 = st.columns(2)
            x1.download_button("⬇️ Burgenliste (CSV)", tdf[cols].to_csv(index=False).encode("utf-8"),
                               f"burgen_{lvl:+.0f}m.csv", "text/csv", use_container_width=True)
            x2.download_button("⬇️ Burgen mit Zustand (GeoJSON)", castles_geojson(tdf.drop(columns=["Wikipedia"])),
                               f"burgen_{lvl:+.0f}m.geojson", "application/geo+json", use_container_width=True)
        st.caption("Für Burgen gilt immer das Badewannen-Modell. Höhendaten sind geglättet – Gipfelburgen erscheinen "
                   "oft einige Meter zu tief. Burgen ohne „castle_type“ in OSM können vereinzelt Schlösser sein.")

# ---------------- Profil ----------------
with t_prof:
    if not S.drawn_line:
        st.info("In der Karte oben links das **Linien-Werkzeug** wählen, Punkte setzen, mit Klick auf den letzten "
                "Punkt beenden. Tipp: vorher „Klick-Abfrage aktiv“ ausschalten.")
    else:
        try:
            res = sample_profile(tuple(tuple(p) for p in S.drawn_line))
        except Exception as e:
            res = None
            st.error(f"Profil konnte nicht berechnet werden: {e}")
        if res:
            df, zp = res
            e = df.hoehe_m.interpolate().bfill().ffill().to_numpy()
            dist = df.distanz_km.to_numpy()
            span = float(np.nanmax(e) - np.nanmin(e)) or 10.0
            base = min(float(np.nanmin(e)), S.level) - max(10.0, span * 0.1)
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=dist, y=np.full_like(e, base), mode="lines", line=dict(width=0),
                                     showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=dist, y=e, fill="tonexty", fillcolor="rgba(139,90,43,0.55)",
                                     line=dict(color="#6b4423"), name="Gelände",
                                     hovertemplate="%{x:.2f} km · %{y:.1f} m<extra></extra>"))
            fig.add_trace(go.Scatter(x=dist, y=np.maximum(S.level, e), fill="tonexty",
                                     fillcolor="rgba(31,120,209,0.45)", line=dict(color="rgba(0,0,0,0)"),
                                     name="Wasser", hoverinfo="skip"))
            fig.add_hline(y=S.level, line_dash="dash", line_color="#1f6fb2",
                          annotation_text=f"Wasserstand {de_num(S.level, 0, True)} m")
            if level2 is not None:
                fig.add_hline(y=level2, line_dash="dot", line_color="#dc2828",
                              annotation_text=f"Vergleich {de_num(level2, 0, True)} m")
            fig.update_layout(height=460, margin=dict(t=30, b=10), xaxis_title="Distanz (km)", yaxis_title="Höhe (m)",
                              legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)
            p = st.columns(4)
            p[0].metric("Länge", f"{de_num(dist[-1], 2)} km")
            p[1].metric("Tiefster Punkt", f"{de_num(np.nanmin(e), 1, True)} m")
            p[2].metric("Höchster Punkt", f"{de_num(np.nanmax(e), 1, True)} m")
            p[3].metric("Anteil unter Wasser", f"{de_num((e < S.level).mean() * 100, 0)} %")
            st.caption(f"Höhendaten mit Kachelzoom {zp}. Profil als Badewanne.")
            b1, b2 = st.columns(2)
            b1.download_button("⬇️ Profil als CSV", df.to_csv(index=False).encode("utf-8"), "profil.csv", "text/csv")
            if b2.button("🗑️ Linie verwerfen"):
                S.drawn_line = None
                st.rerun()

# ---------------- Strandlinien ----------------
with t_terr:
    st.caption("Pro Höhenstufe wird gezählt, wie oft Hangknicke im Ausschnitt vorkommen (1,0 = Durchschnitt).")
    c1, c2, c3 = st.columns([1, 1, 2])
    bin_m = c1.number_input("Höhenschritt (m)", 0.5, 200.0, 5.0, 0.5, key="terr_bin")
    if c2.button("🔍 Ausschnitt analysieren", type="primary"):
        S.terr = dict(tiles=tiles, bin=float(bin_m), df=terrace_signal(tiles, float(bin_m)).to_dict("list"))
    if S.terr:
        cur = pd.DataFrame(S.terr["df"])
        with c3.form("area_form", clear_on_submit=True):
            aname = st.text_input("Name für dieses Gebiet", value=f"Gebiet {len(S.areas) + 1}")
            if st.form_submit_button("📌 Gebiet merken"):
                S.areas[aname] = S.terr["df"]
                st.rerun()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=cur.hoehe_m, y=cur.kantendichte, name="Aktueller Ausschnitt",
                             marker_color="darkorange", opacity=0.75))
        for name, dd in S.areas.items():
            dd = pd.DataFrame(dd)
            fig.add_trace(go.Scatter(x=dd.hoehe_m, y=dd.kantendichte, mode="lines", name=name))
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
        fig.add_vline(x=S.level, line_dash="dash", line_color="#1f6fb2")
        fig.update_layout(height=420, margin=dict(t=30, b=10), xaxis_title="Höhe (m)",
                          yaxis_title="Hangknicke × Durchschnitt", bargap=0, legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, use_container_width=True)
        top = find_peaks(cur.kantendichte.to_numpy())
        if top:
            st.write("**Auffällige Höhen** – Klick setzt den Wasserstand:")
            pc = st.columns(len(top))
            for j, i in enumerate(top):
                h = float(cur.hoehe_m.iloc[i])
                pc[j].button(f"{h:.0f} m · ×{cur.kantendichte.iloc[i]:.1f}", key=f"pk_{j}", on_click=set_level,
                             args=(h,), use_container_width=True)
        if S.areas:
            dsel = st.multiselect("Gemerkte Gebiete entfernen", list(S.areas))
            if dsel and st.button("Entfernen"):
                for n in dsel:
                    S.areas.pop(n, None)
                st.rerun()
        st.caption("⚠️ Auch Ackerterrassen, Straßen oder Gesteinsschichten erzeugen Kanten. Aussagekräftig wird es, "
                   "wenn dieselbe Höhe in mehreren, weit entfernten Gebieten auftaucht.")

# ---------------- Animation ----------------
with t_anim:
    a1, a2, a3, a4 = st.columns(4)
    a1.number_input("Von (m)", LEVEL_MIN, LEVEL_MAX, step=10.0, key="anim_lo")
    a2.number_input("Bis (m)", LEVEL_MIN, LEVEL_MAX, step=10.0, key="anim_hi")
    a3.slider("Bilder", 6, 40, key="anim_steps")
    a4.select_slider("Tempo", list(ANIM_SPEEDS), key="anim_speed")
    b1, b2, b3, b4 = st.columns([2, 2, 2, 1])
    b1.selectbox("Hintergrund", list(BASEMAPS_ANIM), key="anim_base")
    b2.slider("Relief darüberlegen", 0.0, 1.0, step=0.05, key="anim_hill", disabled=S.anim_base == "Nur Relief")
    b3.selectbox("Bildqualität", list(ANIM_QUALITY), key="anim_q")
    b4.write("")
    b4.button("± 100 m", on_click=anim_around, use_container_width=True, help="Bereich ± 100 m um den aktuellen Stand")
    spec = (tiles, float(S.anim_lo), float(S.anim_hi), int(S.anim_steps), S.flood_mode, seed_rc,
            round(float(S.alpha), 2), ANIM_SPEEDS[S.anim_speed], S.anim_base, float(S.anim_hill), S.mode,
            bool(S.shore), ANIM_QUALITY[S.anim_q])
    if st.button("🎬 Animation für den aktuellen Ausschnitt erzeugen", type="primary"):
        if S.anim_hi <= S.anim_lo:
            st.warning("„Bis“ muss größer sein als „Von“.")
        else:
            try:
                frames, gif = make_animation(*spec)
                S.anim = dict(frames=frames, gif=gif, spec=spec)
            except Exception as e:
                st.error(f"Animation konnte nicht erzeugt werden: {e}")
    if S.anim:
        if S.anim["spec"] != spec:
            st.info("Einstellungen oder Ausschnitt wurden geändert – für eine aktuelle Animation neu erzeugen.")
        anim_player(S.anim["frames"], S.anim["spec"][7])
        c1, c2, c3 = st.columns([2, 1, 1])
        pick = c1.select_slider("Wasserstand aus der Animation übernehmen",
                                options=[round(lv, 1) for lv, _ in S.anim["frames"]], key="anim_pick")
        c2.button("✔ Übernehmen", on_click=set_level, args=(float(pick),), use_container_width=True)
        c3.download_button("⬇️ GIF", S.anim["gif"], "flutung.gif", "image/gif", use_container_width=True)
        st.caption("▶/⏸ startet und hält an, ⏮ ⏭ und Schieberegler gehen Bild für Bild. GIF: max. 256 Farben.")

# ---------------- Markierungen ----------------
with t_mark:
    if S.markers:
        st.dataframe(pd.DataFrame(S.markers), use_container_width=True, hide_index=True)
        dsel = st.multiselect("Markierungen löschen", [m["name"] for m in S.markers])
        c1, c2 = st.columns(2)
        if dsel and c1.button("🗑️ Ausgewählte löschen"):
            S.markers = [m for m in S.markers if m["name"] not in dsel]
            st.rerun()
        c2.download_button("⬇️ Markierungen als GeoJSON", markers_geojson(), "markierungen.geojson",
                           "application/geo+json")
    else:
        st.info("Noch keine Markierungen. Klick auf die Karte und dann „⭐ Als Markierung speichern“.")
    up_m = st.file_uploader("Markierungen laden (GeoJSON)", type=["geojson", "json"], key="marker_up")
    if up_m is not None and st.button("Laden"):
        try:
            for f in json.loads(up_m.getvalue()).get("features", []):
                lon_, lat_ = f["geometry"]["coordinates"][:2]
                p = f.get("properties") or {}
                S.markers.append(dict(name=p.get("name", f"Punkt {len(S.markers) + 1}"), notiz=p.get("notiz", ""),
                                      lat=float(lat_), lon=float(lon_), hoehe_m=p.get("hoehe_m"),
                                      wasserstand_m=p.get("wasserstand_m"), zeit=p.get("zeit", "")))
            st.success("Markierungen geladen.")
            st.rerun()
        except Exception as e:
            st.error(f"Datei konnte nicht gelesen werden: {e}")

# ---------------- Export ----------------
with t_exp:
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**🗺️ Karte**")
        if info:
            st.download_button("⬇️ Wasserbild (PNG, transparent)", info["png"], f"wasser_{lvl:+.0f}m.png", "image/png")
            b = info["bounds"]
            st.caption(f"Ausdehnung (S, W, N, O): {b[0]:.4f}, {b[1]:.4f}, {b[2]:.4f}, {b[3]:.4f} · Web-Mercator")
        try:
            sg = shoreline_geojson(tiles, lvl, S.flood_mode, seed_rc)
            if sg:
                st.download_button("⬇️ Uferlinie (GeoJSON, WGS84)", sg, f"uferlinie_{lvl:+.0f}m.geojson",
                                   "application/geo+json")
        except Exception as e:
            st.caption(f"Uferlinie nicht verfügbar: {e}")
    with e2:
        st.markdown("**📈 Daten**")
        try:
            st.download_button("⬇️ Flutkurve (CSV)", flood_curve(tiles).to_csv(index=False).encode("utf-8"),
                               "flutkurve.csv", "text/csv")
        except Exception:
            pass
        if S.terr:
            st.download_button("⬇️ Strandlinien-Analyse (CSV)",
                               pd.DataFrame(S.terr["df"]).to_csv(index=False).encode("utf-8"),
                               "strandlinien.csv", "text/csv")
        if S.markers:
            st.download_button("⬇️ Markierungen (GeoJSON)", markers_geojson(), "markierungen.geojson",
                               "application/geo+json")
    with e3:
        st.markdown("**🔗 Teilen & Sichern**")
        v = S.view or approx_view(S.center, S.zoom)
        st.code(f"?lat={(v[0] + v[2]) / 2:.4f}&lon={(v[1] + v[3]) / 2:.4f}&zoom={v[4]}&level={S.level:.1f}",
                language=None)
        st.caption("Die Adresszeile des Browsers enthält diese Werte – einfach kopieren und verschicken.")
        cfg = {k_: S[k_] for k_ in ("level", "mode", "flood_mode", "alpha", "shore", "hill_on", "band_on", "band_m",
                                    "l2_on", "level2", "sinks_on", "sinks_min")}
        cfg.update(view=list(v), version=APP_VERSION)
        st.download_button("⬇️ Einstellungen (JSON)", json.dumps(cfg, ensure_ascii=False, indent=1).encode(),
                           "flut_einstellungen.json", "application/json")

# ---------------- Hilfe ----------------
with t_help:
    st.markdown("""
### Kurzanleitung
1. **Wasserstand** in der Sidebar einstellen. Minuswerte = tieferer Meeresspiegel.
2. **Karte** verschieben und zoomen – das Wasser wird für den sichtbaren Ausschnitt berechnet.
3. **Flutungsmodus**: Badewanne, vom Kartenrand oder ab angeklicktem Startpunkt.
4. **Klick** auf die Karte: Höhe, Wassertiefe, Überlaufhöhe – als Markierung speicherbar.
5. **3D**: gleiche Wassermaske wie die 2D-Karte, drehbar und zoombar.
6. **Burgen**: Reiter „🏰 Burgen“ → Länder laden → Höhen berechnen → Karte, Uferkurve, Liste. Klick auf eine Burg = Info.
7. **Profil**: Linien-Werkzeug oben links in der Karte, dann Reiter „📈 Profil“.
8. **Strandlinien**: Ausschnitt analysieren, Gebiet merken, an anderem Ort wiederholen.
9. **Animation**: Bereich und Qualität wählen, erzeugen, mit ▶/⏸ abspielen.

### Grenzen & Quellen
- **Badewannen-Modell:** keine Strömung, keine Zeit. Einheitlicher Wasserstand ist eine Vereinfachung; das heutige Gelände ist nicht das damalige.
- **Höhendaten:** AWS Terrain Tiles (SRTM an Land, ETOPO/GEBCO im Meer), an Land ca. 30 m Auflösung, Meeresboden gröber.
- **Burgen:** © OpenStreetMap-Mitwirkende (ODbL) – gezählt wird, was eingetragen ist.
- **Karten:** © OpenStreetMap-Mitwirkende · Esri World Imagery · OpenTopoMap · Ortssuche: Nominatim.
""")

try:
    _v = S.view or approx_view(S.center, S.zoom)
    st.query_params["lat"] = f"{(_v[0] + _v[2]) / 2:.4f}"
    st.query_params["lon"] = f"{(_v[1] + _v[3]) / 2:.4f}"
    st.query_params["zoom"] = str(_v[4])
    st.query_params["level"] = f"{S.level:.1f}"
except Exception:
    pass

