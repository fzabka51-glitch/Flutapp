# =============================================================================
# app.py – Flut-App 0.4 · Weltweite Meeresspiegel- und Flutungssimulation
# Eine Datei. Badewannen-Modell auf Basis der AWS Terrain Tiles (inkl. Meeresboden).
# Werkzeuge: Flutungsmodi, Überlaufhöhe, Senken, Profil, Strandlinien-Vergleich,
# Animation (mit Kartenhintergrund), Markierungen, Overlays, 3D, Export, Link teilen.
# Neu in 0.4: Header-Korrektur, Animation mit Satellit/OSM/Gelände-Hintergrund.
# =============================================================================
from __future__ import annotations

import base64
import io
import json
import math
from datetime import datetime

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from folium import plugins
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from streamlit_folium import st_folium

try:
    import pydeck as pdk
    PYDECK_OK = True
except ImportError:
    PYDECK_OK = False
try:
    from skimage.measure import find_contours
    from skimage.morphology import reconstruction
    SKIMAGE_OK = True
except ImportError:
    SKIMAGE_OK = False

st.set_page_config(page_title="Flut-App", page_icon="🌊", layout="wide")

# =============================================================================
# Konstanten
# =============================================================================
APP_VERSION = "0.4"
LEVEL_MIN, LEVEL_MAX = -11000.0, 5000.0
DEFAULT_CENTER = [50.0, 8.0]
DEFAULT_ZOOM = 5
MAP_HEIGHT = 720
USER_AGENT = "Flut-App (privates Projekt)"

TERRARIUM_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
ESRI_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
OSM_URL = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
TOPO_URL = "https://a.tile.opentopomap.org/{z}/{x}/{y}.png"
MAX_TILE_ZOOM = 15
TILE = 256
EARTH_CIRC = 40075016.686
MAX_LAT = 85.0511
MAX_TILES = 24

# Grobe Kennwerte für die Wasserbilanz (Größenordnungen)
OCEAN_AREA_KM2 = 3.61e8      # Fläche der Ozeane
OCEAN_VOL_KM3 = 1.335e9      # Volumen der Ozeane
ICE_SLE_M = 68.0             # Meeresspiegel-Äquivalent allen Eises, ca. 65–70 m

MODES = ["Wassertiefe", "Veränderung zu heute"]
FLOOD_MODES = ["Badewanne (alles unter dem Wasserstand)",
               "Zusammenhängend vom Kartenrand",
               "Ab Startpunkt (letzter Klick)"]
WATER_CMAP = LinearSegmentedColormap.from_list("wasser", ["#7cc8f8", "#1f78d1", "#08306b"])
COL_NEW = (30, 120, 220)
COL_DRY = (224, 170, 70)
COL_OLD = (150, 190, 230)
COL_SHORE = (255, 215, 0)
COL_L2 = (220, 40, 40)
COL_BAND = (255, 235, 120)

TEXTURES_3D = {
    "Satellit (Esri)": ESRI_URL,
    "OpenStreetMap": OSM_URL,
    "Gelände (OpenTopoMap)": TOPO_URL,
    "Ohne (nur Relief)": None,
}
BASEMAPS_ANIM = {
    "Satellit (Esri)": (ESRI_URL, "© Esri World Imagery"),
    "OpenStreetMap": (OSM_URL, "© OpenStreetMap-Mitwirkende"),
    "Gelände (OpenTopoMap)": (TOPO_URL, "© OpenTopoMap, © OSM"),
    "Nur Relief": (None, ""),
}

PRESETS = [("🌍 Heute", 0.0), ("🧊 Eiszeit −120 m", -120.0), ("🔥 Eis geschmolzen +70 m", 70.0)]
PLACES = {
    "Doggerland, Nordsee (−120 m)": dict(center=[54.5, 3.0], zoom=6, level=-120.0),
    "Beringstraße (−120 m)": dict(center=[65.5, -169.0], zoom=5, level=-120.0),
    "Mittelmeer (−120 m)": dict(center=[38.5, 15.0], zoom=5, level=-120.0),
    "Norddeutschland / Niederlande (+10 m)": dict(center=[52.8, 7.0], zoom=7, level=10.0),
    "Lake Bonneville, Utah (ca. 1550 m)": dict(center=[40.8, -112.8], zoom=7, level=1550.0),
    "Glacial Lake Missoula, Montana (ca. 1280 m)": dict(center=[47.1, -114.2], zoom=7, level=1280.0),
}

S = st.session_state
for _k, _v in dict(
        level=0.0, level_input=0.0, level_slider=0.0, center=DEFAULT_CENTER, zoom=DEFAULT_ZOOM,
        view=None, calc_view=None, mode=MODES[0], flood_mode=FLOOD_MODES[0], alpha=0.65,
        shore=True, auto=True, click=None, click_on=True, place=list(PLACES)[0], _rr=0,
        hill_on=False, hill_op=0.45, band_on=False, band_m=10.0, l2_on=False, level2=100.0,
        sinks_on=False, sinks_min=2.0, wms_url="", wms_layers="", xyz_url="", ovl_op=0.6,
        markers=[], drawn_line=None, areas={}, anim=None, terr=None,
        tex3d=list(TEXTURES_3D)[0], exag3d=3.0, pitch3d=55, bearing3d=0,
        anim_base=list(BASEMAPS_ANIM)[0], anim_hill=0.4).items():
    if _k not in S:
        S[_k] = _v.copy() if isinstance(_v, (list, dict)) else _v

# --- Link-Parameter einmalig übernehmen (?lat=..&lon=..&zoom=..&level=..) ------
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

st.markdown("""
<style>
.block-container {padding-top: 3.2rem; max-width: 1800px;}
.app-header {background: linear-gradient(135deg, #0b3d6b 0%, #1f6fb2 55%, #5fb0e5 100%);
  color: #fff; padding: 16px 24px; border-radius: 14px; margin-bottom: 14px;
  box-shadow: 0 4px 14px rgba(11, 61, 107, .25);}
.app-header h1 {color: #fff !important; margin: 0 !important; padding: 2px 0 !important;
  font-size: 1.8rem; line-height: 1.35 !important; overflow: visible;}
.app-header p {margin: 4px 0 0; opacity: .92;}
div[data-testid="stMetric"] {background: #fff; border: 1px solid #d6e4ef; border-radius: 12px;
  padding: 8px 14px; box-shadow: 0 1px 4px rgba(0, 0, 0, .05);}
.stTabs [data-baseweb="tab-list"] {gap: 6px; flex-wrap: wrap;}
.stTabs [data-baseweb="tab"] {background: #e6f0f7; border-radius: 10px 10px 0 0; padding: 8px 16px;}
.stTabs [aria-selected="true"] {background: #1f6fb2 !important;}
.stTabs [aria-selected="true"] p {color: #fff !important; font-weight: 600;}
.legend span {display: inline-block; width: 14px; height: 14px; border-radius: 3px;
  margin: 0 6px -2px 14px; border: 1px solid #999;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Höhendaten (AWS Terrain Tiles, Terrarium-Format)
# =============================================================================
_session = requests.Session()
_session.headers.update({"User-Agent": USER_AGENT})


def lonlat_to_tile(lon, lat, z):
    n = 2.0 ** z
    lat = np.clip(np.asarray(lat, float), -MAX_LAT, MAX_LAT)
    x = (np.asarray(lon, float) + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n
    return x, y


def tile_to_lonlat(x, y, z):
    n = 2.0 ** z
    lon = np.asarray(x, float) / n * 360.0 - 180.0
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * np.asarray(y, float) / n))))
    return lon, lat


@st.cache_data(show_spinner=False, max_entries=800)
def _fetch_tile(z, x, y):
    r = _session.get(TERRARIUM_URL.format(z=z, x=x, y=y), timeout=20)
    r.raise_for_status()
    a = np.asarray(Image.open(io.BytesIO(r.content)).convert("RGB"), dtype=np.float32)
    return (a[..., 0] * 256.0 + a[..., 1] + a[..., 2] / 256.0 - 32768.0).astype(np.float32)


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
    west, east = west - shift, east - shift
    return south, max(west, -180.0), north, min(east, 180.0)


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
    dem = np.vstack(rows)
    lon_w, lat_n = tile_to_lonlat(tx0, ty0, z)
    lon_e, lat_s = tile_to_lonlat(tx1 + 1, ty1 + 1, z)
    return dem, (float(lat_s), float(lon_w), float(lat_n), float(lon_e)), missing


def pixel_size_m(z):
    return EARTH_CIRC / (TILE * 2 ** z)


def row_px(tiles, nrows):
    """Pixelgröße (m) je Zeile, breitengradabhängig."""
    y = tiles[2] + (np.arange(nrows) + 0.5) / TILE
    lats = tile_to_lonlat(0.0, y, tiles[0])[1]
    return pixel_size_m(tiles[0]) * np.cos(np.radians(lats))


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
    px = min(int((float(x) - tx) * TILE), TILE - 1)
    py = min(int((float(y) - ty) * TILE), TILE - 1)
    v = float(t[py, px])
    return v if np.isfinite(v) else None


# =============================================================================
# Flutung, Analyse, Einfärbung
# =============================================================================
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
    e = mask & ~ndimage.binary_erosion(mask, border_value=1)
    return ndimage.binary_dilation(e)


def water_rgba(dem, water, level, alpha, dref=None):
    a = int(round(255 * alpha))
    d = np.where(water, level - np.nan_to_num(dem), 0.0)
    if dref is None:
        dref = float(np.percentile(d[water], 95)) if water.any() else 1.0
    rgba = WATER_CMAP(np.clip(d / max(dref, 1e-3), 0, 1), bytes=True)
    rgba[..., 3] = np.where(water, a, 0).astype(np.uint8)
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
        below = valid & (dem < level)
        sea_today = valid & (dem < 0)
    new_flood, new_dry = water & ~sea_today, sea_today & ~below
    a = int(round(255 * alpha))
    if mode == MODES[0]:
        rgba = water_rgba(dem, water, level, alpha)
    else:
        rgba = np.zeros(dem.shape + (4,), np.uint8)
        rgba[water & sea_today] = (*COL_OLD, int(a * 0.45))
        rgba[new_flood] = (*COL_NEW, a)
        rgba[new_dry] = (*COL_DRY, a)
    if band:
        with np.errstate(invalid="ignore"):
            zone = valid & ~water & (np.abs(dem - level) <= band)
        rgba[zone] = (*COL_BAND, int(a * 0.7))
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
    volume_km3 = float((depth * ra * 1e6).sum() / 1e9)
    n_bodies = int(ndimage.label(water)[1]) if water.any() else 0
    land = valid & ~water
    lab, n_land = ndimage.label(land)
    n_isl, isl_area = 0, 0.0
    if n_land:
        border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        isl = np.setdiff1d(np.arange(1, n_land + 1), border)
        n_isl = int(isl.size)
        if n_isl:
            isl_area = float(ndimage.sum(np.broadcast_to(ra, dem.shape), lab, isl).sum())
    stats = dict(water=area(water), new_flood=area(new_flood), new_dry=area(new_dry),
                 volume=volume_km3, bodies=n_bodies, islands=n_isl, isl_area=isl_area)

    hill_png = None
    if hill:
        hs = hillshade(dem, row_px(tiles, dem.shape[0]))
        g = (hs * 255).astype(np.uint8)
        hill_png = to_png_bytes(np.dstack([g, g, g, np.where(valid, 255, 0).astype(np.uint8)]))
    return dict(png=to_png_bytes(rgba), hill_png=hill_png, bounds=bounds, stats=stats, missing=missing)


@st.cache_data(show_spinner="Berechne abflusslose Senken …", max_entries=4)
def filled_dem(tiles):
    """Geländemodell mit vom Kartenrand her aufgefüllten Senken (= Überlaufhöhe je Zelle)."""
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
    ra = row_area_km2(tiles, dem.shape[0])[:, None]
    return to_png_bytes(rgba), float((m * ra).sum())


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
    return pd.DataFrame({"wasserstand_m": levels,
                         "flaeche_km2": np.where(idx > 0, cum[np.maximum(idx - 1, 0)], 0.0)})


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
    thr = np.percentile(curv[valid], 95)
    edges, flat = (curv > thr) & (slope > 1.0) & valid, (slope < 2.0) & valid
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
                      "geometry": {"type": "LineString",
                                   "coordinates": [[round(float(a), 6), round(float(b), 6)]
                                                   for a, b in zip(lon, lat)]}})
    return json.dumps({"type": "FeatureCollection", "features": feats}).encode("utf-8")


# --- Profil ----------------------------------------------------------------------
def haversine(lat1, lon1, lat2, lon2):
    R = 6371008.8
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


@st.cache_data(show_spinner="Berechne Höhenprofil …", max_entries=20)
def sample_profile(coords, n=400):
    pts = np.asarray(coords, float)            # [lon, lat]
    seg = haversine(pts[:-1, 1], pts[:-1, 0], pts[1:, 1], pts[1:, 0])
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(cum[-1])
    if total <= 0:
        return None
    d = np.linspace(0, total, n)
    lon, lat = np.interp(d, cum, pts[:, 0]), np.interp(d, cum, pts[:, 1])
    step = total / n
    z = int(np.clip(math.floor(math.log2(EARTH_CIRC * math.cos(math.radians(float(lat.mean())))
                                         / (TILE * max(step, 1.0)))), 0, MAX_TILE_ZOOM))
    elev = [elevation_at(float(a), float(b), z) for a, b in zip(lat, lon)]
    df = pd.DataFrame({"distanz_km": d / 1000, "lat": lat, "lon": lon,
                       "hoehe_m": pd.Series(elev, dtype=float)})
    return df, z


# --- Animation -------------------------------------------------------------------
def _font(size):
    try:
        return ImageFont.truetype(fm.findfont("DejaVu Sans"), size)
    except Exception:
        return ImageFont.load_default()


@st.cache_data(show_spinner=False, max_entries=800)
def _fetch_base_tile(url, z, x, y):
    r = _session.get(url.format(z=z, x=x, y=y), timeout=20)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGB")
    if img.size != (TILE, TILE):
        img = img.resize((TILE, TILE))
    return np.asarray(img, dtype=np.uint8)


def base_mosaic(url, z, tx0, ty0, tx1, ty1):
    """Kartenkacheln für exakt denselben Ausschnitt wie das Höhenmodell."""
    n = 2 ** z
    rows = []
    for ty in range(ty0, ty1 + 1):
        row = []
        for tx in range(tx0, tx1 + 1):
            try:
                t = _fetch_base_tile(url, z, tx % n, ty)
            except Exception:
                t = np.full((TILE, TILE, 3), 210, np.uint8)
            row.append(t)
        rows.append(np.hstack(row))
    return np.vstack(rows)


@st.cache_data(show_spinner="Erzeuge Animation …", max_entries=3)
def make_animation(tiles, lo, hi, steps, flood_mode, seed_rc, alpha, duration_ms,
                   basemap, hill_mix, max_dim=900):
    dem, _, _ = load_mosaic(*tiles)
    s = max(1, math.ceil(max(dem.shape) / max_dim))
    d = dem[::s, ::s]
    hs = hillshade(d, row_px(tiles, dem.shape[0])[::s] * s)
    url, attribution = BASEMAPS_ANIM.get(basemap, (None, ""))
    if url:
        bm = base_mosaic(url, *tiles)[::s, ::s][:d.shape[0], :d.shape[1]].astype(np.float32) / 255.0
        shade = (1.0 - hill_mix) + hill_mix * (0.45 + 0.8 * hs[..., None])
        base = np.clip(bm * shade, 0, 1)
    else:
        base = np.dstack([(hs * 190 + 50)] * 3).astype(np.float32) / 255.0
    seed = (seed_rc[0] // s, seed_rc[1] // s) if seed_rc else None
    dref = max(hi - lo, 1.0)
    font = _font(max(14, d.shape[1] // 32))
    small = _font(max(10, d.shape[1] // 70))
    frames, pils = [], []
    for lv in np.linspace(lo, hi, steps):
        w = compute_water(d, lv, flood_mode, seed)
        top = water_rgba(d, w, lv, alpha, dref).astype(np.float32) / 255.0
        ta = top[..., 3:4]
        rgb = ((top[..., :3] * ta + base * (1 - ta)) * 255 + 0.5).astype(np.uint8)
        img = Image.fromarray(rgb, "RGB")
        draw = ImageDraw.Draw(img)
        txt = f"Wasserstand {lv:+.0f} m"
        bx = draw.textbbox((0, 0), txt, font=font)
        draw.rectangle([10, 10, 26 + bx[2] - bx[0], 26 + bx[3] - bx[1]], fill=(30, 30, 30))
        draw.text((18 - bx[0], 18 - bx[1]), txt, fill=(255, 255, 255), font=font)
        if attribution:
            ab = draw.textbbox((0, 0), attribution, font=small)
            x0, y0 = img.width - (ab[2] - ab[0]) - 12, img.height - (ab[3] - ab[1]) - 10
            draw.rectangle([x0 - 4, y0 - 3, img.width - 6, img.height - 5], fill=(255, 255, 255))
            draw.text((x0 - ab[0], y0 - ab[1]), attribution, fill=(60, 60, 60), font=small)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        frames.append((float(lv), buf.getvalue()))
        pils.append(img.quantize(colors=255))
    gif = io.BytesIO()
    durs = [int(duration_ms)] * (len(pils) - 1) + [int(duration_ms) * 4]
    pils[0].save(gif, format="GIF", save_all=True, append_images=pils[1:], duration=durs, loop=0)
    return frames, gif.getvalue()


# --- 3D ------------------------------------------------------------------------------
def build_deck(lat, lon, zoom, level, exag, alpha, texture, pitch, bearing):
    dec = {"rScaler": 256 * exag, "gScaler": 1 * exag, "bScaler": exag / 256, "offset": -32768 * exag}
    kw = dict(elevation_decoder=dec, elevation_data=TERRARIUM_URL, max_zoom=15,
              wireframe=False, pickable=False)
    tex_url = TEXTURES_3D.get(texture)
    if tex_url:
        kw["texture"] = tex_url
    else:
        kw["color"] = [200, 190, 170]
    terrain = pdk.Layer("TerrainLayer", **kw)
    span = float(np.clip(360.0 / 2 ** zoom * 3, 0.5, 80.0))
    h = level * exag
    s, n = max(lat - span, -85.0), min(lat + span, 85.0)
    w, e = lon - span * 2, lon + span * 2
    water = pdk.Layer("PolygonLayer", data=[{"polygon": [[w, s, h], [e, s, h], [e, n, h], [w, n, h]]}],
                      get_polygon="polygon", get_fill_color=[30, 120, 220, int(255 * alpha)],
                      stroked=False, filled=True, extruded=False, pickable=False)
    view = pdk.ViewState(latitude=lat, longitude=lon, zoom=zoom, pitch=pitch, bearing=bearing, max_pitch=85)
    return pdk.Deck(layers=[terrain, water], initial_view_state=view, map_style=None)


# =============================================================================
# Hilfsfunktionen & Callbacks
# =============================================================================
def de_num(v, nd=1, sign=False):
    s = f"{v:{'+' if sign else ''},.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")


def approx_view(center, zoom, w=1200, h=MAP_HEIGHT):
    x, y = lonlat_to_tile(center[1], center[0], zoom)
    dx, dy = w / 2 / TILE, h / 2 / TILE
    lon_w, lat_n = tile_to_lonlat(x - dx, y - dy, zoom)
    lon_e, lat_s = tile_to_lonlat(x + dx, y + dy, zoom)
    return (round(float(lat_s), 4), round(float(lon_w), 4), round(float(lat_n), 4),
            round(float(lon_e), 4), int(zoom))


def view_from(out):
    if not out or not out.get("bounds"):
        return None
    sw, ne = out["bounds"].get("_southWest"), out["bounds"].get("_northEast")
    if not sw or not ne or sw.get("lat") is None or ne.get("lat") is None:
        return None
    z = out.get("zoom") or S.zoom
    return (round(sw["lat"], 4), round(sw["lng"], 4), round(ne["lat"], 4), round(ne["lng"], 4), int(z))


def set_level(v):
    v = float(np.clip(v, LEVEL_MIN, LEVEL_MAX))
    S.level, S.level_input, S.level_slider = v, v, v


def shift_level(d):
    set_level(S.level + d)


def go_to(center, zoom, level=None):
    S.center, S.zoom = [float(center[0]), float(center[1])], int(zoom)
    S.view = approx_view(S.center, S.zoom)
    S.calc_view = S.view
    if level is not None:
        set_level(level)


def calc_now():
    S.calc_view = S.view


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
    """Grundkarte mit Werkzeugen – bleibt gleich, damit sie nicht neu geladen wird."""
    m = folium.Map(location=DEFAULT_CENTER, zoom_start=DEFAULT_ZOOM, tiles=None,
                   world_copy_jump=True, control_scale=True, min_zoom=2)
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
    folium.TileLayer(tiles=ESRI_URL, attr="Esri World Imagery", name="Satellit (Esri)").add_to(m)
    folium.TileLayer(tiles="https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png",
                     attr="© OpenTopoMap (CC-BY-SA), © OpenStreetMap-Mitwirkende",
                     name="Gelände (OpenTopoMap)", max_zoom=17).add_to(m)
    plugins.Fullscreen(position="topleft").add_to(m)
    plugins.MeasureControl(position="topleft", primary_length_unit="kilometers",
                           primary_area_unit="sqkilometers").add_to(m)
    plugins.Draw(export=False, position="topleft",
                 draw_options={"polyline": {"shapeOptions": {"color": "#e41a1c", "weight": 3}},
                               "polygon": False, "rectangle": False, "circle": False,
                               "marker": False, "circlemarker": False},
                 edit_options={"edit": False, "remove": True}).add_to(m)
    plugins.MousePosition(position="bottomleft", separator=" · ", num_digits=4, prefix="Lat/Lon:").add_to(m)
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)
    folium.LayerControl(position="topright").add_to(m)
    return m


# =============================================================================
# Sidebar
# =============================================================================
with st.sidebar:
    st.markdown("## 🌊 Wasserstand")
    st.metric("Aktueller Wasserstand", f"{de_num(S.level, sign=True)} m")
    st.slider("Grobeinstellung (m)", LEVEL_MIN, LEVEL_MAX, step=10.0, key="level_slider",
              on_change=lambda: set_level(S.level_slider))
    st.number_input("Genauer Wert (m)", LEVEL_MIN, LEVEL_MAX, step=1.0, format="%.1f",
                    key="level_input", on_change=lambda: set_level(S.level_input))
    bc = st.columns(6)
    for i, d in enumerate([-100, -10, -1, 1, 10, 100]):
        bc[i].button(f"{d:+d}", key=f"lv_{d}", on_click=shift_level, args=(float(d),),
                     use_container_width=True)
    for label, lv in PRESETS:
        st.button(label, key=f"pre_{lv}", on_click=set_level, args=(lv,), use_container_width=True)

    with st.expander("🧭 Flutungsmodus", expanded=True):
        st.radio("Wie breitet sich das Wasser aus?", FLOOD_MODES, key="flood_mode",
                 help="Badewanne: jede tiefe Stelle wird blau. Vom Kartenrand: nur Wasser, das mit dem Rand "
                      "des Ausschnitts verbunden ist (abgeschlossene Becken bleiben trocken). Ab Startpunkt: "
                      "nur das Becken des zuletzt angeklickten Punkts.")
        st.checkbox("Klick-Abfrage aktiv", key="click_on",
                    help="Zum Zeichnen von Profillinien vorübergehend ausschalten.")

    with st.expander("🎨 Darstellung"):
        st.radio("Ansicht", MODES, key="mode")
        st.slider("Deckkraft Wasser", 0.2, 1.0, step=0.05, key="alpha")
        st.checkbox("Uferlinie", key="shore")
        st.checkbox("Relief-Schummerung", key="hill_on")
        st.slider("Deckkraft Relief", 0.1, 1.0, step=0.05, key="hill_op", disabled=not S.hill_on)
        st.checkbox("Abflusslose Senken (lila)", key="sinks_on", disabled=not SKIMAGE_OK,
                    help="Becken ohne Abfluss zum Kartenrand – dort könnten sich Seen bilden.")
        st.slider("Mindesttiefe Senken (m)", 0.5, 50.0, step=0.5, key="sinks_min", disabled=not S.sinks_on)
        st.checkbox("Automatisch aktualisieren", key="auto")
        if not S.auto:
            st.button("🔄 Ausschnitt berechnen", on_click=calc_now, use_container_width=True, type="primary")

    with st.expander("⚖️ Vergleich & Unsicherheit"):
        st.checkbox("Zweiter Wasserstand als rote Linie", key="l2_on")
        st.number_input("Zweiter Wasserstand (m)", LEVEL_MIN, LEVEL_MAX, step=10.0, key="level2",
                        disabled=not S.l2_on)
        st.checkbox("Unsicherheitszone anzeigen (gelb)", key="band_on",
                    help="Land, das innerhalb ± x m um den Wasserstand liegt.")
        st.slider("± Meter", 1.0, 200.0, step=1.0, key="band_m", disabled=not S.band_on)

    with st.expander("🧩 Eigene Kartenebenen"):
        st.text_input("WMS-URL", key="wms_url", placeholder="https://…/wms?",
                      help="z. B. geologische Karte eines Landesamts")
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


# =============================================================================
# Hauptbereich – gemeinsame Berechnungen
# =============================================================================
st.markdown("""<div class="app-header"><h1>🌊 Flut-App</h1>
<p>Meeresspiegel und Wasserstände weltweit simulieren – von −11 000 m bis +5000 m</p></div>""",
            unsafe_allow_html=True)

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

tabs = st.tabs(["🗺️ Karte", "🏔️ 3D", "📈 Profil", "🏖️ Strandlinien", "🎬 Animation",
                "📍 Markierungen", "📦 Export", "❓ Hilfe"])
t_map, t_3d, t_prof, t_terr, t_anim, t_mark, t_exp, t_help = tabs

# =============================================================================
# Karte
# =============================================================================
with t_map:
    fg = folium.FeatureGroup(name="Flut-Ebenen")
    info = None
    try:
        info = water_layer(tiles, lvl, S.mode, S.flood_mode, seed_rc, round(float(S.alpha), 2),
                           bool(S.shore), level2, band, bool(S.hill_on))
        ib = info["bounds"]
        box = [[ib[0], ib[1]], [ib[2], ib[3]]]
        if info["hill_png"]:
            folium.raster_layers.ImageOverlay(to_uri(info["hill_png"]), bounds=box, opacity=float(S.hill_op),
                                              interactive=False, zindex=5).add_to(fg)
        if S.sinks_on and SKIMAGE_OK:
            spng, _ = sinks_layer(tiles, float(S.sinks_min))
            folium.raster_layers.ImageOverlay(to_uri(spng), bounds=box, opacity=1.0,
                                              interactive=False, zindex=8).add_to(fg)
        folium.raster_layers.ImageOverlay(to_uri(info["png"]), bounds=box, opacity=1.0,
                                          interactive=False, zindex=10).add_to(fg)
    except Exception as e:
        st.error(f"Höhendaten konnten nicht geladen werden: {e}")

    if S.wms_url.strip() and S.wms_layers.strip():
        folium.raster_layers.WmsTileLayer(url=S.wms_url.strip(), layers=S.wms_layers.strip(), fmt="image/png",
                                          transparent=True, name="WMS", overlay=True,
                                          opacity=float(S.ovl_op)).add_to(fg)
    if S.xyz_url.strip():
        folium.TileLayer(tiles=S.xyz_url.strip(), attr="Eigene Ebene", name="XYZ", overlay=True,
                         opacity=float(S.ovl_op)).add_to(fg)
    for mk in S.markers:
        folium.Marker([mk["lat"], mk["lon"]], tooltip=mk["name"],
                      popup=folium.Popup(f"<b>{mk['name']}</b><br>{mk.get('notiz', '')}<br>"
                                         f"Höhe: {mk.get('hoehe_m', '–')} m", max_width=260),
                      icon=folium.Icon(color="green", icon="star")).add_to(fg)
    if S.click:
        folium.Marker([S.click["lat"], S.click["lon"]], tooltip="Abfragepunkt",
                      icon=folium.Icon(color="red", icon="info-sign")).add_to(fg)
    if S.drawn_line:
        folium.PolyLine([[b, a] for a, b in S.drawn_line], color="#e41a1c", weight=3,
                        tooltip="Profillinie").add_to(fg)

    out = st_folium(build_map(), key="karte", height=MAP_HEIGHT, use_container_width=True,
                    center=S.center, zoom=S.zoom, feature_group_to_add=fg,
                    returned_objects=["bounds", "zoom", "last_clicked", "all_drawings"])

    need_rerun = False
    new_view = view_from(out)
    if new_view and new_view != S.view:
        S.view = new_view
        if S.auto and choose_tiles(normalize_bounds(*new_view[:4]), new_view[4]) != tiles:
            need_rerun = True

    drw = (out or {}).get("all_drawings") or []
    lines = [f for f in drw if f and (f.get("geometry") or {}).get("type") == "LineString"]
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

    # --- Legende ---
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
    st.markdown(f"<div class='legend'>{leg}</div>", unsafe_allow_html=True)

    # --- Kennzahlen ---
    if info:
        stt = info["stats"]
        lat_c = (bounds[0] + bounds[2]) / 2
        res_m = pixel_size_m(tiles[0]) * math.cos(math.radians(lat_c))
        r1 = st.columns(4)
        r1[0].metric("Wasserstand", f"{de_num(S.level, sign=True)} m")
        r1[1].metric("Wasserfläche im Ausschnitt", f"{de_num(stt['water'], 0)} km²")
        r1[2].metric("Neu überflutet", f"{de_num(stt['new_flood'], 0)} km²")
        r1[3].metric("Trockengefallen", f"{de_num(stt['new_dry'], 0)} km²")
        r2 = st.columns(4)
        r2[0].metric("Wasservolumen im Ausschnitt", f"{de_num(stt['volume'], 1)} km³")
        r2[1].metric("Getrennte Wasserflächen", f"{stt['bodies']:,}".replace(",", "."))
        r2[2].metric("Inseln", f"{stt['islands']:,}".replace(",", "."),
                     f"{de_num(stt['isl_area'], 0)} km²", delta_color="off")
        r2[3].metric("Auflösung Höhendaten", f"≈ {de_num(res_m, 0)} m", f"Kachelzoom {tiles[0]}",
                     delta_color="off")
        if info["missing"]:
            st.warning(f"{info['missing']} Höhenkachel(n) fehlen – dort ist kein Wasser dargestellt.")
        if S.flood_mode == FLOOD_MODES[2] and not S.click:
            st.info("Modus „Ab Startpunkt“: Klicke auf die Karte, um den Startpunkt zu setzen.")

    # --- Klick-Abfrage ---
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
                            k[3].metric("Überlaufhöhe", "kein Becken", "Wasser fließt zum Rand ab",
                                        delta_color="off")
                except Exception:
                    pass
        st.caption("Überlaufhöhe: bis zu diesem Wasserstand kann hier ein abgeschlossener See stehen, darüber "
                   "läuft er zum Rand des Ausschnitts über. Hängt vom gewählten Kartenausschnitt ab.")
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
        st.caption("💡 Klick auf die Karte = Höhe, Wassertiefe und Überlaufhöhe. Werkzeuge oben links: "
                   "Vollbild, Messen, Linie zeichnen (für das Profil). Ebenen-Symbol oben rechts = Hintergrund.")

    with st.expander("💧 Wasserbilanz – wie viel Wasser bräuchte dieser Meeresspiegel?"):
        dv = OCEAN_AREA_KM2 * S.level / 1000.0
        w1, w2, w3 = st.columns(3)
        w1.metric("Volumenänderung weltweit (grob)", f"{de_num(dv / 1e6, 1, True)} Mio. km³")
        w2.metric("Anteil am heutigen Ozeanvolumen", f"{de_num(dv / OCEAN_VOL_KM3 * 100, 2, True)} %")
        w3.metric("Vielfaches allen Eises", f"{de_num(S.level / ICE_SLE_M, 1, True)} ×")
        st.caption(f"Überschlag: Ozeanfläche ({de_num(OCEAN_AREA_KM2 / 1e6, 0)} Mio. km²) × Wasserstand. "
                   f"Alles Eis der Erde entspricht ca. 65–70 m Meeresspiegel. Vernachlässigt: überflutete "
                   "Landflächen, Absenkung/Hebung des Untergrunds, Wärmeausdehnung. Nur Größenordnung.")

    with st.expander("📈 Flutkurve für den sichtbaren Ausschnitt"):
        try:
            fc = flood_curve(tiles)
            if len(fc):
                f = go.Figure(go.Scatter(x=fc.wasserstand_m, y=fc.flaeche_km2, mode="lines",
                                         line=dict(color="#1f6fb2"), name="Fläche"))
                f.add_vline(x=S.level, line_dash="dash", line_color="gray")
                f.update_layout(height=340, margin=dict(t=20, b=10), xaxis_title="Wasserstand (m)",
                                yaxis_title="Überflutete Fläche (km²)")
                st.plotly_chart(f, use_container_width=True)
                st.caption("Steile Stufen = Höhen, ab denen große Ebenen oder Becken volllaufen (Badewanne).")
        except Exception as e:
            st.warning(f"Flutkurve nicht verfügbar: {e}")

# =============================================================================
# 3D
# =============================================================================
with t_3d:
    if not PYDECK_OK:
        st.error("Für die 3D-Ansicht wird 'pydeck' benötigt.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        tex = c1.selectbox("Oberfläche", list(TEXTURES_3D), key="tex3d")
        exag = c2.slider("Überhöhung", 1.0, 20.0, step=0.5, key="exag3d")
        pitch = c3.slider("Neigung (°)", 0, 85, key="pitch3d")
        bearing = c4.slider("Drehung (°)", -180, 180, step=5, key="bearing3d")
        v = S.view or approx_view(S.center, S.zoom)
        deck = build_deck((v[0] + v[2]) / 2, (v[1] + v[3]) / 2, max(v[4] - 0.5, 1.0), float(S.level),
                          float(exag), float(S.alpha) * 0.85, tex, int(pitch), int(bearing))
        try:
            st.pydeck_chart(deck, use_container_width=True, height=MAP_HEIGHT)
        except TypeError:
            st.pydeck_chart(deck, use_container_width=True)
        st.caption("🖱️ Mausrad = Zoom · Linke Maustaste = Verschieben · Rechte Maustaste (oder Strg + links) = "
                   f"Kippen/Drehen. Wasserebene bei {de_num(S.level, sign=True)} m (Badewanne).")

# =============================================================================
# Profil
# =============================================================================
with t_prof:
    if not S.drawn_line:
        st.info("So geht's: In der Karte oben links das **Linien-Werkzeug** wählen, Punkte setzen und mit "
                "Klick auf den letzten Punkt (oder „Finish“) beenden. Tipp: vorher in der Sidebar "
                "„Klick-Abfrage aktiv“ ausschalten.")
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
            fig.update_layout(height=460, margin=dict(t=30, b=10), xaxis_title="Distanz (km)",
                              yaxis_title="Höhe (m)", legend=dict(orientation="h", y=1.08))
            st.plotly_chart(fig, use_container_width=True)
            wet = e < S.level
            p = st.columns(4)
            p[0].metric("Länge", f"{de_num(dist[-1], 2)} km")
            p[1].metric("Tiefster Punkt", f"{de_num(np.nanmin(e), 1, True)} m")
            p[2].metric("Höchster Punkt", f"{de_num(np.nanmax(e), 1, True)} m")
            p[3].metric("Anteil unter Wasser", f"{de_num(wet.mean() * 100, 0)} %")
            st.caption(f"Höhendaten mit Kachelzoom {zp}. Profil über Badewanne – ohne Flutungsmodus.")
            b1, b2 = st.columns(2)
            b1.download_button("⬇️ Profil als CSV", df.to_csv(index=False).encode("utf-8"), "profil.csv", "text/csv")
            if b2.button("🗑️ Linie verwerfen"):
                S.drawn_line = None
                st.rerun()

# =============================================================================
# Strandlinien
# =============================================================================
with t_terr:
    st.caption("Stand Wasser lange auf einer Höhe, entstehen oft Terrassen und Kanten – alle auf **derselben "
               "Höhe**. Hier wird pro Höhenstufe gezählt, wie oft Hangknicke im Ausschnitt vorkommen "
               "(1,0 = Durchschnitt). Mit „Gebiet merken“ kannst du mehrere Regionen vergleichen.")
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
        for name, d in S.areas.items():
            dd = pd.DataFrame(d)
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
                pc[j].button(f"{h:.0f} m · ×{cur.kantendichte.iloc[i]:.1f}", key=f"pk_{j}",
                             on_click=set_level, args=(h,), use_container_width=True)
        if S.areas:
            dsel = st.multiselect("Gemerkte Gebiete entfernen", list(S.areas))
            if dsel and st.button("Entfernen"):
                for n in dsel:
                    S.areas.pop(n, None)
                st.rerun()
        st.caption("⚠️ Nur ein Hinweis: Auch Ackerterrassen, Straßen oder Gesteinsschichten erzeugen Kanten auf "
                   "gleicher Höhe. Aussagekräftig wird es, wenn dieselbe Höhe in **mehreren, weit entfernten "
                   "Gebieten** auftaucht. Grobe Höhendaten (große Ausschnitte) verwischen feine Terrassen.")

# =============================================================================
# Animation
# =============================================================================
with t_anim:
    a1, a2, a3, a4 = st.columns(4)
    lo = a1.number_input("Von (m)", LEVEL_MIN, LEVEL_MAX, float(max(LEVEL_MIN, S.level - 100)), 10.0, key="anim_lo")
    hi = a2.number_input("Bis (m)", LEVEL_MIN, LEVEL_MAX, float(min(LEVEL_MAX, S.level + 100)), 10.0, key="anim_hi")
    steps = a3.slider("Bilder", 6, 40, 16, key="anim_steps")
    speed = a4.select_slider("Tempo", ["langsam", "mittel", "schnell"], value="mittel", key="anim_speed")
    b1, b2 = st.columns(2)
    basemap = b1.selectbox("Hintergrund", list(BASEMAPS_ANIM), key="anim_base")
    hill_mix = b2.slider("Relief darüberlegen", 0.0, 1.0, step=0.05, key="anim_hill",
                         disabled=basemap == "Nur Relief",
                         help="0 = reine Karte · 1 = Karte stark plastisch schattiert")
    dur = {"langsam": 700, "mittel": 400, "schnell": 200}[speed]
    if st.button("🎬 Animation für den aktuellen Ausschnitt erzeugen", type="primary"):
        if hi <= lo:
            st.warning("„Bis“ muss größer sein als „Von“.")
        else:
            frames, gif = make_animation(tiles, float(lo), float(hi), int(steps), S.flood_mode, seed_rc,
                                         float(S.alpha), dur, basemap, float(hill_mix))
            S.anim = dict(frames=frames, gif=gif)
            S.anim_idx = 0
    if S.anim:
        g1, g2 = st.columns(2)
        with g1:
            st.markdown("**▶ Film (läuft automatisch)**")
            st.image(S.anim["gif"], use_container_width=True)
            st.download_button("⬇️ GIF herunterladen", S.anim["gif"], "flutung.gif", "image/gif")
        with g2:
            st.markdown("**⏸ Bild für Bild (angehalten)**")
            n = len(S.anim["frames"])
            if S.get("anim_idx", 0) > n - 1:
                S.anim_idx = 0
            i = st.slider("Bild", 0, n - 1, key="anim_idx",
                          help="Schieberegler bewegen = vor- und zurückspulen")
            lv_i, png_i = S.anim["frames"][min(i, n - 1)]
            st.image(png_i, caption=f"Wasserstand {de_num(lv_i, 0, True)} m", use_container_width=True)
            st.button("Diesen Wasserstand übernehmen", on_click=set_level, args=(lv_i,))
        st.caption("Die Animation zeigt den aktuellen Kartenausschnitt mit dem Flutungsmodus aus der Sidebar. "
                   "Nach Änderung von Hintergrund oder Ausschnitt bitte neu erzeugen.")

# =============================================================================
# Markierungen
# =============================================================================
with t_mark:
    if S.markers:
        mdf = pd.DataFrame(S.markers)
        st.dataframe(mdf, use_container_width=True, hide_index=True)
        dsel = st.multiselect("Markierungen löschen", [m["name"] for m in S.markers])
        c1, c2 = st.columns(2)
        if dsel and c1.button("🗑️ Ausgewählte löschen"):
            S.markers = [m for m in S.markers if m["name"] not in dsel]
            st.rerun()
        c2.download_button("⬇️ Markierungen als GeoJSON", markers_geojson(), "markierungen.geojson",
                           "application/geo+json")
    else:
        st.info("Noch keine Markierungen. Klick auf die Karte und dann „⭐ Als Markierung speichern“.")
    up = st.file_uploader("Markierungen laden (GeoJSON)", type=["geojson", "json"])
    if up is not None and st.button("Laden"):
        try:
            for f in json.loads(up.getvalue()).get("features", []):
                lon_, lat_ = f["geometry"]["coordinates"][:2]
                p = f.get("properties") or {}
                S.markers.append(dict(name=p.get("name", f"Punkt {len(S.markers) + 1}"), notiz=p.get("notiz", ""),
                                      lat=float(lat_), lon=float(lon_), hoehe_m=p.get("hoehe_m"),
                                      wasserstand_m=p.get("wasserstand_m"), zeit=p.get("zeit", "")))
            st.success("Markierungen geladen.")
            st.rerun()
        except Exception as e:
            st.error(f"Datei konnte nicht gelesen werden: {e}")

# =============================================================================
# Export
# =============================================================================
with t_exp:
    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**🗺️ Karte**")
        if info:
            st.download_button("⬇️ Wasserbild (PNG, transparent)", info["png"],
                               f"wasser_{lvl:+.0f}m.png", "image/png")
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
        params = dict(lat=f"{(v[0] + v[2]) / 2:.4f}", lon=f"{(v[1] + v[3]) / 2:.4f}", zoom=str(v[4]),
                      level=f"{S.level:.1f}")
        st.code("?" + "&".join(f"{k}={val}" for k, val in params.items()), language=None)
        st.caption("Die Adresszeile des Browsers enthält diese Werte – einfach kopieren und verschicken.")
        cfg = {k: S[k] for k in ("level", "mode", "flood_mode", "alpha", "shore", "hill_on", "band_on", "band_m",
                                 "l2_on", "level2", "sinks_on", "sinks_min")}
        cfg.update(view=list(v), version=APP_VERSION)
        st.download_button("⬇️ Einstellungen (JSON)", json.dumps(cfg, ensure_ascii=False, indent=1).encode(),
                           "flut_einstellungen.json", "application/json")

# =============================================================================
# Hilfe
# =============================================================================
with t_help:
    st.markdown("""
### Kurzanleitung
1. **Wasserstand** in der Sidebar einstellen (Regler, Zahl oder ±-Knöpfe). Minuswerte zeigen tiefere Meeresspiegel.
2. **Karte** verschieben und zoomen – das Wasser wird für den sichtbaren Ausschnitt berechnet.
3. **Flutungsmodus** wählen: Badewanne, vom Kartenrand oder ab einem angeklickten Startpunkt.
4. **Klick** auf die Karte: Höhe, Wassertiefe, Überlaufhöhe – und als Markierung speichern.
5. **Profil:** Linien-Werkzeug oben links in der Karte, dann Reiter „📈 Profil“.
6. **Strandlinien:** Ausschnitt analysieren, Gebiet merken, an anderem Ort wiederholen und vergleichen.
7. **Animation:** Bereich und Hintergrund wählen, erzeugen, als Film ansehen oder Bild für Bild durchgehen.

### Kartenwerkzeuge (oben links)
⛶ Vollbild · 📏 Messen (Strecken/Flächen) · ✏️ Linie zeichnen · 🗑️ Linie löschen · unten links Koordinaten.

### Tipps
- Kleine Ausschnitte = feinere Höhendaten. Für Täler und Terrassen hineinzoomen.
- „Veränderung zu heute“ + Minuswert zeigt trockengefallenen Meeresboden in Ocker.
- Bei langsamer Reaktion „Automatisch aktualisieren“ ausschalten und per Knopf rechnen.
""")
    with st.expander("ℹ️ Grenzen des Modells & Datenquellen", expanded=True):
        st.markdown("""
- **Badewannen-Modell:** keine Strömung, keine Zeit. Im Modus Badewanne werden auch Senken ohne Meeresverbindung
  blau (z. B. Kaspisches Becken, Totes Meer).
- **„Vom Kartenrand“ und Überlaufhöhe** beziehen sich auf den **sichtbaren Ausschnitt** – ein anderer Ausschnitt
  kann andere Ergebnisse liefern.
- **Einheitlicher Wasserstand:** Frühere Meeresspiegel waren regional unterschiedlich (Hebung/Senkung durch Eislast).
  Das heutige Gelände ist nicht das damalige (Erosion, Ablagerung, Bebauung).
- **Höhendaten:** AWS Terrain Tiles (Mapzen; u. a. SRTM an Land, ETOPO/GEBCO im Meer). An Land bei starkem Zoom ca.
  30 m Auflösung, Fehler von mehreren Metern möglich. **Meeresboden deutlich gröber.** Bezug ≈ heutiger Meeresspiegel.
- **Wasserbilanz:** grobe Überschlagsrechnung, nur Größenordnung.
- **Karten:** © OpenStreetMap-Mitwirkende · Esri World Imagery · OpenTopoMap · Ortssuche: Nominatim (OSM).
- Seehöhen der Beispielorte sind gerundet – für genaue Werte Fachliteratur nutzen.
""")

# --- Link in der Adresszeile aktuell halten -------------------------------------------
try:
    _v = S.view or approx_view(S.center, S.zoom)
    st.query_params["lat"] = f"{(_v[0] + _v[2]) / 2:.4f}"
    st.query_params["lon"] = f"{(_v[1] + _v[3]) / 2:.4f}"
    st.query_params["zoom"] = str(_v[4])
    st.query_params["level"] = f"{S.level:.1f}"
except Exception:
    pass

